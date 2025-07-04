"""
Script to analyze math evaluation results and suggest tools through LLM reflection.
Maximum of 10 tools will be maintained through consolidation and prioritization.
Each tool will include a Python function implementation.
"""

import json
import random
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from api_client import create_openai_client, make_api_call

# Example tools that already exist
EXISTING_TOOLS = []

REFLECTION_PROMPT = """You are a tool design expert. Given a math problem and its solution attempt, suggest tools that could help solve similar problems more effectively.

CRITICAL CONSTRAINT: The total number of tools (including existing ones) must not exceed 20. Currently there are {num_existing} existing tools.

Make sure your tools are not too specific. They should be general enough to be useful for multiple similar problems.

Here are the existing tools available:
{existing_tools}

Analyze the following problem and solution attempt:

Problem: {problem}

Student's Solution Attempt:
{solution}

Correct Answer: {correct_answer}
Was Correct: {was_correct}

Based on this example, reflect on:
1. What mathematical concepts and steps were needed?
2. What tools could have helped solve this more effectively?
3. Are the existing tools sufficient? If not, what new tool would you suggest?
4. Could any existing tools be merged or replaced to maintain the 20-tool limit?

IMPORTANT:
- Suggest only ONE new tool that would be most impactful
- If suggesting a new tool would exceed 10 tools total, you MUST suggest which existing tools to merge or replace
- Focus on tools that are general enough to help with multiple similar problems
- You MUST provide a Python async function implementation for the tool
- The Python implementation should be complete and functional
- Use standard Python libraries (math, sympy, numpy) as needed
- Make sure to keep the imports minimal and all of them inside the function definition

Format your tool suggestion as a JSON object with these fields:
{{
    "tool_name": "string",
    "description": "string",
    "parameters": {{
        "type": "object",
        "properties": {{
            "param_name": {{
                "type": "string",
                "description": "string"
            }}
        }},
        "required": ["param_name"]
    }},
    "python_implementation": "string",
    "required_imports": ["string"],
    "replaces_tools": ["string"],
    "merges_with_tools": ["string"],
    "priority_score": 0,
    "generality_score": 0
}}

Example Python Implementation:
```python
async def tool_name(param1: str, param2: float) -> Dict:
    \"\"\"Tool description\"\"\"
    # Implementation here
    result = await some_calculation(param1, param2)
    return {{
        "result": result,
        "steps": ["step1", "step2"]  # Optional steps for transparency
    }}
```

Focus on creating tools that are:
1. Reusable across similar problems
2. Specific enough to be implementable
3. General enough to handle variations
4. Complementary to existing tools
5. Actually implementable with standard Python libraries

Provide your reflection and then list your tool suggestion in valid JSON format."""

async def analyze_sample(
    client,
    sample: Dict,
    existing_tools: List[Dict],
    temperature: float = 0.7
) -> Dict:
    """Analyze a single sample and get tool suggestions"""
    
    # Format existing tools for prompt
    tools_str = json.dumps(existing_tools, indent=2)
    num_existing = len(existing_tools)
    
    # Create messages for reflection
    messages = [{
        "role": "user",
        "content": REFLECTION_PROMPT.format(
            num_existing=num_existing,
            existing_tools=tools_str,
            problem=sample["problem"],
            solution=sample["response"],
            correct_answer=sample["gold_answer"],
            was_correct=sample["is_correct"]
        )
    }]
    
    # Get reflection and suggestions
    response = await make_api_call(client, messages, temperature)
    reflection = response.choices[0].message.content
    
    # Try to extract JSON tool suggestions from the response
    try:
        # Find JSON-like blocks in the response
        json_start = reflection.find("{")
        json_end = reflection.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = reflection[json_start:json_end]
            tool_suggestion = json.loads(json_str)
            
            # Validate Python implementation
            if "python_implementation" in tool_suggestion:
                # Basic validation that it's an async function
                impl = tool_suggestion["python_implementation"]
                if not "async def" in impl:
                    tool_suggestion["validation_error"] = "Implementation must be an async function"
        else:
            tool_suggestion = None
    except json.JSONDecodeError:
        tool_suggestion = None
    
    return {
        "timestamp": datetime.now().isoformat(),
        "problem_id": sample["unique_id"],
        "problem": sample["problem"],
        "was_correct": sample["is_correct"],
        "reflection": reflection,
        "tool_suggestion": tool_suggestion
    }

def consolidate_tools(reflections: List[Dict], existing_tools: List[Dict]) -> List[Dict]:
    """Consolidate tool suggestions to maintain maximum of 10 tools"""
    
    # Extract all valid tool suggestions
    all_tools = []
    for reflection in reflections:
        if reflection.get("tool_suggestion") and not reflection.get("tool_suggestion", {}).get("validation_error"):
            tool = reflection["tool_suggestion"]
            # Add source problem info
            tool["source_problem"] = {
                "id": reflection["problem_id"],
                "problem": reflection["problem"],
                "was_correct": reflection["was_correct"]
            }
            all_tools.append(tool)
    
    # Sort tools by priority and generality
    all_tools.sort(key=lambda x: (
        x.get("priority_score", 0) + x.get("generality_score", 0),
        x.get("priority_score", 0)
    ), reverse=True)
    
    # Process tools that want to merge with existing ones
    merged_tools = []
    remaining_tools = []
    existing_tool_names = {t["function"]["name"] for t in existing_tools}
    
    for tool in all_tools:
        merge_targets = tool.get("merges_with_tools", [])
        if merge_targets and any(t in existing_tool_names for t in merge_targets):
            merged_tools.append(tool)
        else:
            remaining_tools.append(tool)
    
    # Calculate how many new tools we can add
    max_new_tools = 10 - len(existing_tools) + len(merged_tools)
    
    # Select top tools up to the limit
    selected_tools = remaining_tools[:max_new_tools]
    
    return merged_tools + selected_tools

def extract_tool_schema(tool: Dict) -> Dict:
    """Extract the tool schema and definition from a tool suggestion"""
    return {
        "type": "function",
        "function": {
            "name": tool["tool_name"],
            "description": tool["description"],
            "parameters": tool["parameters"]
        },
        "python_implementation": tool["python_implementation"].strip(),
        "required_imports": tool.get("required_imports", [])
    }

async def main():
    # Load evaluation results
    results_file = "simple_eval_20250629_171054_EN-EASY_0-100.jsonl"
    samples = []
    
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    print(f"📊 Loaded {len(samples)} valid samples")
    
    # Initialize OpenAI client
    client = create_openai_client()
    
    # Process all samples in batches of 5
    batch_size = 10
    all_reflections = []
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        print(f"\n🤔 Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size} (samples {i+1}-{min(i+batch_size, len(samples))})")
        
        # Analyze batch
        tasks = [analyze_sample(client, sample, EXISTING_TOOLS) for sample in batch]
        batch_reflections = await asyncio.gather(*tasks)
        all_reflections.extend(batch_reflections)
        
        # Print interim summary
        print(f"✅ Processed {len(batch)} samples in this batch")
        print(f"📝 Total reflections so far: {len(all_reflections)}")
        
        # Optional: Add a small delay between batches to avoid rate limits
        if i + batch_size < len(samples):
            await asyncio.sleep(1)
    
    # Consolidate tools from all reflections
    consolidated_tools = consolidate_tools(all_reflections, EXISTING_TOOLS)
    
    # Save tool schemas and implementations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tools_file = f"math_tools_{timestamp}.jsonl"
    
    with open(tools_file, "w", encoding="utf-8") as f:
        # Save existing tools first
        for tool in EXISTING_TOOLS:
            f.write(json.dumps(tool, ensure_ascii=False) + "\n")
        
        # Save new consolidated tools
        for tool in consolidated_tools:
            if "tool_name" in tool and "parameters" in tool:
                tool_schema = extract_tool_schema(tool)
                f.write(json.dumps(tool_schema, ensure_ascii=False) + "\n")
    
    # Generate Python module with all tools
    module_file = f"math_tools_{timestamp}.py"
    with open(module_file, "w", encoding="utf-8") as f:
        f.write('"""Auto-generated math tools module"""\n\n')
        
        # Collect all required imports
        all_imports = set()
        for tool in consolidated_tools:
            all_imports.update(tool.get("required_imports", []))
        
        # Write imports
        for imp in sorted(all_imports):
            f.write(f"import {imp}\n")
        f.write("\n\n")
        
        # Write tool implementations
        for tool in consolidated_tools:
            if "python_implementation" in tool:
                f.write(tool["python_implementation"].strip() + "\n\n")
    
    print(f"\n📝 Tool schemas and implementations saved to: {tools_file}")
    print(f"📝 Python implementations saved to: {module_file}")
    
    # Print summary
    print("\n🛠️ Tools Summary:")
    print(f"Existing tools: {len(EXISTING_TOOLS)}")
    print(f"New/merged tools: {len(consolidated_tools)}")
    print(f"Total tools: {len(EXISTING_TOOLS) + len(consolidated_tools)}")
    
    print("\nTool Details:")
    for tool in consolidated_tools:
        print(f"\nTool: {tool['tool_name']}")
        print(f"Description: {tool['description']}")
        if tool.get('required_imports'):
            print(f"Required imports: {', '.join(tool['required_imports'])}")

if __name__ == "__main__":
    asyncio.run(main()) 