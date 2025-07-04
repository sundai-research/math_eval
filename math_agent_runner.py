"""
Math problem solver agent runner with tool-based answer submission.
Uses asynchronous API calls for concurrent math problem solving.
"""

import os
import json
import asyncio
import time
import re
from math_tools import MATH_SUBMIT_TOOL_SCHEMA, execute_math_tool_call
from api_client import make_api_call, create_openai_client
from typing import List, Dict, Union, Optional
from tenacity import RetryError
from datetime import datetime
from math_verification import extract_boxed, verify_answer

from tool_call_engine import ToolDefinition, ToolSchema, ToolManager


async def solve_math_problem_async(
    unique_id: str,
    sample_idx: int,
    messages: List[Dict[str, str]],
    ground_truth: str,
    tool_manager: ToolManager,
    max_iterations: int = 3,
    verbose: bool = True,
    writer=None,  # AsyncJSONLWriter instance
    temperature: float = 0.6,
    semaphore: Optional[asyncio.Semaphore] = None,
    problem_data: Optional[Dict] = None,  # Full problem data for JSONL output
) -> Dict[str, Union[bool, int, List[Dict], str]]:
    """
    Asynchronously solve a math problem with tool-based answer submission.

    Args:
        unique_id: Problem identifier
        sample_idx: Sample number for this problem
        messages: List of message dictionaries in OpenAI format
        ground_truth: The correct answer content (without \\boxed{})
        max_iterations: Maximum number of attempts allowed (default: 3)
        max_tool_retries: Maximum number of tool call retry attempts (default: 3)
        verbose: Whether to print detailed output (default: True)
        writer: Optional AsyncJSONLWriter to write results as they complete
        temperature: Sampling temperature for the model
        semaphore: Optional semaphore to limit concurrent API calls

    Returns:
        Dict containing:
            - success: bool - True if the problem was solved correctly
            - iterations_used: int - Number of iterations used
            - conversation: List[Dict] - Full conversation history
            - final_status: str - Description of how the problem ended
            - unique_id: str - Problem identifier
            - sample_idx: int - Sample number
            - tool_attempts: int - Number of tool calls made
            - final_submitted_answer: str - Last answer submitted via tool
            - extracted_final_answer: str - Content extracted from final answer
    """

    # Helper function to create result dict and optionally write to JSONL
    async def create_and_write_result(
        success: bool,
        iterations: int,
        status: str,
        conversation: List[Dict],
        tool_attempts: int = 0,
        final_answer: str = "",
        extracted_answer: str = "",
    ):
        result = {
            "success": success,
            "iterations_used": iterations,
            "conversation": conversation,
            "final_status": status,
            "unique_id": unique_id,
            "sample_idx": sample_idx,
            "tool_attempts": tool_attempts,
            "final_submitted_answer": final_answer,
            "extracted_final_answer": extracted_answer,
        }

        # Write to JSONL if writer is provided
        if writer:
            # Calculate total tokens from conversation (approximate)
            total_content_length = sum(
                len(msg.get("content", ""))
                for msg in conversation
                if msg.get("content")
            )

            jsonl_result = {
                "unique_id": unique_id,
                "sample_idx": sample_idx,
                "success": success,
                "iterations_used": iterations,
                "final_status": status,
                "tool_attempts": tool_attempts,
                "conversation_length": len(conversation),
                "timestamp": datetime.now().isoformat(),
                "final_submitted_answer": final_answer,
                "extracted_final_answer": extracted_answer,
                "extracted_answer": extracted_answer,  # For compatibility with analyze command
                "is_correct": success,  # For compatibility with analysis
                "gold_answer": ground_truth,  # For compatibility with analyze command
                "ground_truth": ground_truth,
                "full_conversation": conversation,
                "response_length": total_content_length,  # Approximate
                "output_tokens": total_content_length // 4,  # Rough estimate
            }

            # Add problem data if provided
            if problem_data:
                jsonl_result["problem"] = problem_data.get("problem", "")
                jsonl_result["subject"] = problem_data.get("subject", "unknown")
            await writer.write(jsonl_result)

        return result

    # Wrapper to reduce repetition - always uses current state
    async def return_result(success: bool, status: str):
        return await create_and_write_result(
            success=success,
            iterations=iteration,
            status=status,
            conversation=conversation_messages.copy(),
            tool_attempts=tool_call_count,
            final_answer=last_submitted_answer,
            extracted_answer=last_extracted_answer,
        )

    verbose = True

    # Use semaphore for concurrency control
    async with semaphore:
        # Initialize variables
        iteration = 0
        tool_call_count = 0
        last_submitted_answer = ""
        last_extracted_answer = ""

        # Check if API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            if verbose:
                print(f"Problem {unique_id}-{sample_idx}: OPENAI_API_KEY not set")
            return await return_result(False, "OPENAI_API_KEY not set")

        # openai client
        client = create_openai_client()

        # Define the tools available to the model
        tools = [
            {"type": t["type"], "function": t["function"]}
            for t in tool_manager.read_tools()
        ]
        tools += [MATH_SUBMIT_TOOL_SCHEMA]
        # tools += [ENUMERATE_SMALL_CASES_TOOL_SCHEMA]

        # Copy messages to avoid modifying the original
        conversation_messages = messages.copy()

        if verbose:
            print(f"Problem {unique_id}-{sample_idx}: Starting math problem solving")

        start_time = time.time()

        while iteration < max_iterations:
            iteration += 1

            if verbose:
                print(
                    f"Problem {unique_id}-{sample_idx}: Iteration {iteration}/{max_iterations}"
                )

            # Call the model asynchronously with retry logic
            try:
                response = await make_api_call(
                    client, conversation_messages, temperature, tools=tools
                )
            except RetryError as e:
                elapsed_time = time.time() - start_time
                if verbose:
                    print(
                        f"Problem {unique_id}-{sample_idx}: ⏰ Timeout after {elapsed_time:.1f}s"
                    )
                    print(f"  💥 Last exception: {str(e.last_attempt.exception())}")
                    import traceback

                    print(f"  📍 Traceback: {traceback.format_exc()}")
                return await return_result(
                    False,
                    f"Retry timeout after {elapsed_time:.1f} seconds - {str(e.last_attempt.exception())}",
                )
            except Exception as e:
                if verbose:
                    print(
                        f"Problem {unique_id}-{sample_idx}: 💥 Unexpected API error: {e}"
                    )
                    import traceback

                    print(f"  📍 Traceback: {traceback.format_exc()}")
                return await return_result(False, f"Unexpected error: {str(e)}")

            # Get the response
            assistant_message = response.choices[0].message

            # Parse answer from response content (similar to process_sample)
            if assistant_message.content:
                # Extract answer from content
                boxed_answers = extract_boxed(assistant_message.content)
                extracted = boxed_answers[0] if boxed_answers else ""

                # Verify answer directly if found
                if extracted:
                    is_correct = verify_answer(extracted, ground_truth)

                    if is_correct:
                        # Update tracking variables
                        last_submitted_answer = assistant_message.content
                        last_extracted_answer = extracted

                        if verbose:
                            print(
                                f"Problem {unique_id}-{sample_idx}: ✅ Found correct answer in response text: {extracted}"
                            )
                            print(
                                f"Problem {unique_id}-{sample_idx}: Solved in {iteration} iterations (via text parsing)"
                            )

                        # Add the response to conversation
                        conversation_messages.append(
                            {"role": "assistant", "content": assistant_message.content}
                        )

                        return await return_result(
                            True, "Problem solved successfully via text parsing"
                        )

            if verbose:
                # Enhanced logging for better pattern detection
                if assistant_message.content:
                    content = assistant_message.content
                    content_preview = content.replace("\n", " ")[:300]
                    print(
                        f"Problem {unique_id}-{sample_idx}: 📄 Model response preview: {content_preview}..."
                    )

                    # Detailed pattern analysis
                    patterns_found = []

                    # Check for answer patterns in text
                    if "\\boxed{" in content:
                        patterns_found.append("\\boxed{} in text")
                        # Extract what's in the boxed to see what model tried to submit
                        boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", content)
                        if boxed_matches:
                            print(
                                f"  📦 Found \\boxed{{}} content in text: {boxed_matches}"
                            )

                    # Check for code patterns
                    if "import " in content or "def " in content or "print(" in content:
                        patterns_found.append("Python code")

                    # Check for calculation patterns
                    if "=" in content and any(
                        op in content for op in ["+", "-", "*", "/", "^"]
                    ):
                        patterns_found.append("Calculations")

                    # Check for solution indicators
                    if any(
                        phrase in content.lower()
                        for phrase in [
                            "the answer is",
                            "therefore",
                            "thus",
                            "so the",
                            "final answer",
                        ]
                    ):
                        patterns_found.append("Solution phrases")

                    # Check for step-by-step working
                    if "Step " in content or "step " in content:
                        patterns_found.append("Step-by-step working")

                    # Report patterns
                    if patterns_found:
                        print(f"  🔍 Patterns detected: {', '.join(patterns_found)}")

                    # Count response length indicators
                    word_count = len(content.split())
                    line_count = content.count("\n") + 1
                    print(f"  📏 Response size: {word_count} words, {line_count} lines")

                else:
                    print(
                        f"Problem {unique_id}-{sample_idx}: 📄 Model response: [No content]"
                    )

                if assistant_message.tool_calls:
                    print(
                        f"Problem {unique_id}-{sample_idx}: 🔧 Model made {len(assistant_message.tool_calls)} tool call(s)"
                    )
                    # Detailed tool call logging
                    for i, tc in enumerate(assistant_message.tool_calls):
                        args_preview = tc.function.arguments[:200]
                        print(f"  📞 Tool call {i + 1}: {tc.function.name}")
                        print(
                            f"     Args: {args_preview}{'...' if len(tc.function.arguments) > 200 else ''}"
                        )

                        # Check for patterns in tool arguments
                        if (
                            "import" in tc.function.arguments
                            or "print" in tc.function.arguments
                        ):
                            print(f"     ⚠️  WARNING: Tool args contain code patterns!")
                        if "\\\\boxed{" in tc.function.arguments:
                            print(f"     ✓  Tool args contain \\\\boxed{{}} notation")
                else:
                    print(
                        f"Problem {unique_id}-{sample_idx}: ❗ Model made NO tool calls"
                    )
                    if iteration == 1:
                        print(f"  🚨 CRITICAL: Failed to use tool on first attempt!")

            # Check if the model wants to use tools
            if assistant_message.tool_calls:
                if verbose:
                    print(
                        f"Problem {unique_id}-{sample_idx}: Submitting answer via tool"
                    )

                for tool_call in assistant_message.tool_calls:
                    tool_call_count += 1
                    function_name = tool_call.function.name

                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        if verbose:
                            print(
                                f"Problem {unique_id}-{sample_idx}: 💥 Failed to parse tool arguments: {e}"
                            )
                            print(f"  📄 Raw arguments: {tool_call.function.arguments}")
                        tool_result = (
                            f"❌ ERROR: Failed to parse tool arguments: {str(e)}"
                        )
                        function_args = {}
                        last_submitted_answer = ""
                    else:
                        # Store the submitted answer for tracking
                        print(f"loaded function arguments {function_args=}")
                        last_submitted_answer = function_args.get("answer", "")

                        if verbose:
                            print(
                                f"Problem {unique_id}-{sample_idx}: 🔧 Calling tool '{function_name}'"
                            )

                        # Execute the tool (now async)
                        tool_result = execute_math_tool_call(
                            function_name,
                            function_args,
                            ground_truth,
                            tool_manager=tool_manager,
                            verbose=verbose,
                        )

                    boxed_content = extract_boxed(last_submitted_answer)

                    if boxed_content:
                        last_extracted_answer = boxed_content[0]
                    else:
                        # Use the same flexible extraction as in math_tools.py
                        last_extracted_answer = last_submitted_answer.strip()

                        # Remove common prefixes/suffixes
                        if last_extracted_answer.startswith("The answer is "):
                            last_extracted_answer = last_extracted_answer[14:].strip()
                        if last_extracted_answer.startswith("Answer: "):
                            last_extracted_answer = last_extracted_answer[8:].strip()
                        if last_extracted_answer.endswith("."):
                            last_extracted_answer = last_extracted_answer[:-1].strip()

                        # Remove quotes if present
                        if (
                            len(last_extracted_answer) > 2
                            and last_extracted_answer[0] == '"'
                            and last_extracted_answer[-1] == '"'
                        ):
                            last_extracted_answer = last_extracted_answer[1:-1]

                    # Add the tool call and result to messages
                    conversation_messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call.model_dump()],
                        }
                    )

                    conversation_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        }
                    )

                    # Check if solution was correct
                    if "CORRECT!" in tool_result:
                        if verbose:
                            print(
                                f"Problem {unique_id}-{sample_idx}: Solved in {iteration} iterations"
                            )
                        return await return_result(True, "Problem solved successfully")

            else:
                # Model provided a regular response
                conversation_messages.append(
                    {"role": "assistant", "content": assistant_message.content}
                )

        if verbose:
            print(
                f"Problem {unique_id}-{sample_idx}: ❌ Failed after {max_iterations} iterations"
            )
            print(f"  📊 Total tool attempts made: {tool_call_count}")
            print(
                f"  💬 Final conversation length: {len(conversation_messages)} messages"
            )

            # Analyze failure patterns
            failure_analysis = []
            boxed_in_text_count = 0
            code_submission_count = 0

            for msg in conversation_messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    content = msg["content"]
                    if "\\boxed{" in content:
                        boxed_in_text_count += 1
                    if "import " in content or "def " in content:
                        code_submission_count += 1

            if tool_call_count == 0:
                failure_analysis.append("Never used tool")
            if boxed_in_text_count > 0:
                failure_analysis.append(
                    f"Put \\boxed{{}} in text {boxed_in_text_count} times"
                )
            if code_submission_count > 0:
                failure_analysis.append(f"Wrote code {code_submission_count} times")
            if last_submitted_answer and "import" in last_submitted_answer:
                failure_analysis.append("Submitted code as answer")

            if failure_analysis:
                print(f"  ⚠️  Failure analysis: {'; '.join(failure_analysis)}")

            # Show last submitted answer if any
            if last_submitted_answer:
                print(
                    f"  📝 Last submitted: '{last_submitted_answer[:100]}{'...' if len(last_submitted_answer) > 100 else ''}'"
                )
                print(f"  🎯 Expected: '{ground_truth}'")

        return await return_result(
            False, f"Reached maximum iterations ({max_iterations}) without solving"
        )


def create_math_tool_system_prompt() -> str:
    """Create the system prompt for math problem solving with tools."""
    return """You are an expert mathematician tasked with solving mathematical problems step by step.

You have access to one tool:
- submit_math_answer: Use this to submit your final answer

IMPORTANT INSTRUCTIONS:
1. Work through the problem step by step, showing your reasoning
2. When you have your final answer, use the submit_math_answer tool
3. Your answer MUST be wrapped in \\boxed{} notation in the tool call
4. If your answer is incorrect, you'll receive feedback and can try again
5. You can make multiple attempts to get the correct answer

Examples of proper tool usage:
- For a number: submit_math_answer(answer="\\boxed{42}")
- For an expression: submit_math_answer(answer="\\boxed{x^2 - 4}")
- For a fraction: submit_math_answer(answer="\\boxed{\\frac{3}{4}}")

Always show your work before submitting your final answer."""
