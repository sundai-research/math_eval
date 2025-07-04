"""
Tool-based Math Evaluator - Uses generated tools from reflection system
"""

import json
import os
import asyncio
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from math_verification import verify_answer_with_math_verify
from api_client import make_api_call, create_openai_client
import typer
from tool_call_engine import ToolManager

app = typer.Typer(help="Tool-based math evaluation using generated tools")

class AsyncJSONLWriter:
    """Simple async JSONL writer"""
    def __init__(self, filename: str):
        self.filename = filename
        self.queue = asyncio.Queue()
        self.file_handle = None
        self.is_running = False
        self._writer_task = None

    async def start(self):
        """Start the writer"""
        if not self.is_running:
            self.file_handle = open(self.filename, "a", encoding="utf-8")
            self.is_running = True
            self._writer_task = asyncio.create_task(self._writer_loop())

    async def _writer_loop(self):
        """Process write requests"""
        while self.is_running or not self.queue.empty():
            try:
                data = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                self.file_handle.write(json.dumps(data, ensure_ascii=False) + "\n")
                self.file_handle.flush()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error writing to JSONL: {e}")

        if self.file_handle:
            self.file_handle.close()

    async def write(self, data: Dict):
        """Queue data for writing"""
        await self.queue.put(data)

    async def stop(self):
        """Stop the writer"""
        self.is_running = False
        if self._writer_task:
            await self._writer_task

def load_completed_attempts(jsonl_file: str) -> Set[Tuple[str, int]]:
    """Load completed attempts from existing JSONL file"""
    completed = set()

    if not os.path.exists(jsonl_file):
        return completed

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                unique_id = data.get("unique_id")
                sample_idx = data.get("sample_idx")
                if unique_id is not None and sample_idx is not None:
                    completed.add((unique_id, sample_idx))
            except json.JSONDecodeError:
                continue

    print(f"📖 Loaded {len(completed)} completed attempts from {jsonl_file}")
    return completed

def format_tool_schema(tools: List[Dict]) -> str:
    """Format tool schemas into a readable string for the prompt"""
    tool_descriptions = []
    for tool in tools:
        func = tool["function"]
        params = func["parameters"]["properties"]
        required = func["parameters"].get("required", [])
        
        param_desc = []
        for name, details in params.items():
            req = "*" if name in required else ""
            param_desc.append(f"  - {name}{req}: {details['type']} - {details['description']}")
        
        tool_descriptions.append(
            f"Tool: {func['name']}\n"
            f"Description: {func['description']}\n"
            f"Parameters:\n" + "\n".join(param_desc)
        )
    
    return "\n\n".join(tool_descriptions)

def parse_tool_call(text: str) -> Optional[Tuple[str, Dict]]:
    """Parse a tool call from the model's response"""
    # Look for patterns like: use tool_name(param1=value1, param2=value2)
    # or: call tool_name(param1=value1, param2=value2)
    tool_pattern = r"(?:use|call)\s+(\w+)\s*\((.*?)\)"
    match = re.search(tool_pattern, text, re.IGNORECASE)
    
    if not match:
        return None
        
    tool_name = match.group(1)
    params_str = match.group(2)
    
    # Parse parameters
    params = {}
    param_pattern = r'(\w+)\s*=\s*([^,)]+)'
    for param_match in re.finditer(param_pattern, params_str):
        name = param_match.group(1)
        value = param_match.group(2).strip()
        
        # Convert value to appropriate type
        try:
            # Try to parse as JSON first
            params[name] = json.loads(value)
        except json.JSONDecodeError:
            # If not JSON, keep as string but remove quotes
            params[name] = value.strip('"\'')
    
    return tool_name, params

def check_for_final_answer(text: str) -> Optional[str]:
    """Check if the response contains a final answer submission"""
    # Look for patterns like: final answer: X or submit answer: X
    patterns = [
        r"final\s+answer\s*:\s*(.+?)(?:\n|$)",
        r"submit\s+answer\s*:\s*(.+?)(?:\n|$)",
        r"the\s+answer\s+is\s*:\s*(.+?)(?:\n|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

async def process_with_tools(
    client,
    problem_data: Dict,
    sample_idx: int,
    tool_manager: ToolManager,
    writer: AsyncJSONLWriter,
    semaphore: asyncio.Semaphore,
    temperature: float,
    max_iterations: int = 3
) -> None:
    """Process a single problem using available tools"""
    async with semaphore:
        try:
            # Get all available tools and format their schemas
            tools = tool_manager.read_tools()
            tool_schemas = format_tool_schema(tools)
            
            # Create system message with tool information
            system_msg = {
                "role": "system",
                "content": f"""You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_schemas}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
            }

            # Create user message with problem
            user_msg = {
                "role": "user",
                "content": f"Solve this math problem step by step:\n\n{problem_data['problem']}"
            }

            messages = [system_msg, user_msg]
            iterations = 0
            final_answer = None
            tool_calls = []
            is_correct = False
            
            while iterations < max_iterations and not final_answer:
                iterations += 1
                
                try:
                    # Get model response
                    response = await make_api_call(
                        client, 
                        messages, 
                        temperature,
                        tools=tools
                    )
                    
                    # Get the response text
                    response_text = response.choices[0].message.content
                    
                    # Add assistant's message to history
                    messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
                    # Check for tool calls
                    tool_call = parse_tool_call(response_text)
                    print(tool_call)
                    if tool_call:
                        tool_name, tool_args = tool_call
                        try:
                            # Execute tool using ToolManager
                            result = tool_manager.execute_tool(tool_name, **tool_args)
                            tool_calls.append({
                                "tool": tool_name,
                                "args": tool_args,
                                "result": result
                            })
                            
                            # Add result to messages
                            messages.append({
                                "role": "user",
                                "content": f"Tool result: {json.dumps(result, indent=2)}"
                            })
                        except Exception as e:
                            error_msg = f"Error using {tool_name}: {str(e)}"
                            messages.append({
                                "role": "user",
                                "content": error_msg
                            })
                    
                    # Check for final answer
                    # answer = check_for_final_answer(response_text)
                    is_correct = verify_answer_with_math_verify(response_text, problem_data["answer"])
                    print(f"is_correct: {is_correct}")
                    
                except Exception as e:
                    print(f"Error in iteration {iterations}: {str(e)}")
                    messages.append({
                        "role": "user",
                        "content": f"Error: {str(e)}. Please try a different approach."
                    })
            
            # Write result
            await writer.write({
                "unique_id": problem_data["unique_id"],
                "sample_idx": sample_idx,
                "problem": problem_data["problem"],
                "gold_answer": problem_data["answer"],
                "response": messages,
                "tool_calls": tool_calls,
                "final_answer": final_answer,
                "is_correct": is_correct,
                "iterations": iterations,
                "timestamp": datetime.now().isoformat()
            })

            status = "CORRECT" if is_correct else "INCORRECT"
            print(f"✓ {problem_data['unique_id']}-{sample_idx}: {status} ({iterations} iterations)")

        except Exception as e:
            print(f"✗ {problem_data['unique_id']}-{sample_idx}: ERROR - {str(e)}")
            await writer.write({
                "unique_id": problem_data["unique_id"],
                "sample_idx": sample_idx,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

async def run_tool_evaluation(
    min_idx: int,
    max_idx: int,
    samples: int,
    temperature: float,
    concurrency: int,
    dataset: str,
    tools_file: str,
    output: Optional[str],
    max_iterations: int
):
    """Main evaluation function"""
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Initialize ToolManager
    print(f"🔧 Loading tools from {tools_file}")
    tool_manager = ToolManager(tools_file)
    tools = tool_manager.read_tools()
    print(f"📊 Loaded {len(tools)} tools")

    # Load dataset
    dataset_file = f"data/OlymMATH-{dataset}.jsonl"
    problems = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))

    # Select problems
    selected = problems[min_idx:max_idx]
    print(f"📊 Loaded {len(selected)} problems from index {min_idx} to {max_idx}")

    # Determine output file
    if output:
        output_file = output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"tool_eval_{timestamp}_{dataset}_{min_idx}-{max_idx}.jsonl"

    # Load completed attempts
    completed_attempts = load_completed_attempts(output_file)

    # Initialize client
    client = create_openai_client()

    # Create writer
    writer = AsyncJSONLWriter(output_file)
    await writer.start()

    # Create semaphore
    semaphore = asyncio.Semaphore(concurrency)

    # Create tasks
    tasks = []
    skipped = 0
    for problem in selected:
        for sample_idx in range(samples):
            if (problem["unique_id"], sample_idx) in completed_attempts:
                print(f"⏭️  Skipping {problem['unique_id']}-{sample_idx} (already completed)")
                skipped += 1
                continue

            task = process_with_tools(
                client,
                problem,
                sample_idx,
                tool_manager,
                writer,
                semaphore,
                temperature,
                max_iterations
            )
            tasks.append(task)

    if skipped > 0:
        print(f"📊 Skipped {skipped} already completed attempts")

    results = []
    if tasks:
        print(f"🏃 Running {len(tasks)} evaluations...")
        results = await asyncio.gather(*tasks)
        print(f"🏃 Completed {len(results)} evaluations")
    else:
        print("⚠️  No new tasks to run (all already completed)")

    # Stop writer
    await writer.stop()

    print(f"\n📄 Results saved to: {output_file}")
    return results

@app.command()
def main(
    min_idx: int = typer.Option(..., "--min", help="Start index for problems (inclusive)"),
    max_idx: int = typer.Option(..., "--max", help="End index for problems (exclusive)"),
    samples: int = typer.Option(10, "--samples", help="Number of samples per question"),
    temperature: float = typer.Option(0.6, "--temperature", help="Sampling temperature"),
    concurrency: int = typer.Option(10, "--concurrency", help="Maximum concurrent API requests"),
    dataset: str = typer.Option("EN-EASY", "--dataset", help="Dataset to use (EN-EASY, EN-HARD, ZH-EASY, ZH-HARD)"),
    tools_file: str = typer.Option(..., "--tools", help="Path to JSONL file containing tool definitions"),
    output: Optional[str] = typer.Option(None, "--output", help="Output JSONL file (default: auto-generated)"),
    max_iterations: int = typer.Option(3, "--max-iterations", help="Maximum attempts per problem")
):
    """Run tool-based math evaluation using tools from JSONL file"""
    asyncio.run(run_tool_evaluation(
        min_idx,
        max_idx,
        samples,
        temperature,
        concurrency,
        dataset,
        tools_file,
        output,
        max_iterations
    ))

if __name__ == "__main__":
    app() 