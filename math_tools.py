"""
Math problem solving tools for interactive answer submission.
Contains tool schemas and execution functions for math answer verification.
"""
from math_verification import extract_boxed, verify_answer


def submit_math_answer(answer: str, ground_truth: str, verbose: bool = True) -> str:
    """
    Evaluate a math answer submission against the ground truth.
    
    Args:
        answer: String containing the answer (with or without \\boxed{} notation)
        ground_truth: The correct answer content (without \\boxed{})
        verbose: Whether to print detailed logging
        
    Returns:
        String indicating whether the answer is correct or not
    """
    try:
        # First try to extract from \\boxed{} if present
        boxed_answers = extract_boxed(answer)
        
        if boxed_answers:
            # Found \\boxed{} content
            extracted_content = boxed_answers[0]
        else:
            # No \\boxed{} found, use the answer directly
            # Strip common patterns that might appear
            extracted_content = answer.strip()
            
            # Remove common prefixes/suffixes that models might add
            if extracted_content.startswith("The answer is "):
                extracted_content = extracted_content[14:].strip()
            if extracted_content.startswith("Answer: "):
                extracted_content = extracted_content[8:].strip()
            if extracted_content.endswith("."):
                extracted_content = extracted_content[:-1].strip()
            
            # If it's wrapped in quotes, remove them
            if len(extracted_content) > 2 and extracted_content[0] == '"' and extracted_content[-1] == '"':
                extracted_content = extracted_content[1:-1]
            
            if verbose and not boxed_answers:
                print(f"  📝 Note: Answer submitted without \\boxed{{}}, using raw content")
        
        if verbose:
            print(f"  📝 Submitted answer: '{extracted_content}'")
            print(f"  🎯 Ground truth: '{ground_truth}'")
        
        # Verify against ground truth using existing verification function
        is_correct = verify_answer(extracted_content, ground_truth)
        
        if is_correct:
            result = "✅ CORRECT! Your answer matches the expected solution."
            if verbose:
                print(f"  ✅ Answer verification: CORRECT")
        else:
            result = "❌ INCORRECT. Your answer does not match the expected solution. Please try again."
            if verbose:
                print(f"  ❌ Answer verification: INCORRECT")
        
        return result
            
    except Exception as e:
        error_msg = f"❌ ERROR: Could not process your answer. {str(e)}"
        if verbose:
            print(f"  💥 Exception during tool submission: {str(e)}")
            import traceback
            print(f"  📍 Traceback: {traceback.format_exc()}")
        return error_msg


# Tool schema for the math answer submission tool
MATH_SUBMIT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_math_answer",
        "description": "Submit your final mathematical answer. Your answer will be parsed and compared against the expected solution. While common prefixes may be stripped, including phrases like 'The answer is', 'The volume is', etc. can cause verification to fail. Submit ONLY the mathematical value or expression itself for best results.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The mathematical answer only. Good examples: \"42\", \"x^2 - 4\", \"\\frac{3}{4}\", \"\\boxed{42}\". Bad examples: \"The answer is 42\", \"The volume is \\frac{18}{5}\". Including descriptive text may cause verification to fail."
                }
            },
            "required": ["answer"]
        }
    }
}


def execute_math_tool_call(tool_name: str, arguments: dict, ground_truth: str, verbose: bool = True) -> str:
    """
    Execute a math tool call and return the result.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of tool arguments
        ground_truth: The correct answer for verification
        verbose: Whether to print detailed logging
        
    Returns:
        String result from tool execution
    """
    if tool_name == "submit_math_answer":
        return submit_math_answer(arguments["answer"], ground_truth, verbose)
    else:
        error_msg = f"❌ ERROR: Unknown tool: {tool_name}"
        if verbose:
            print(f"  💥 Tool execution error: {error_msg}")
        return error_msg