"""
Math problem solving tools for interactive answer submission.
Contains tool schemas and execution functions for math answer verification.
"""
from math_verification import extract_boxed, verify_answer
from prompts import format_enumerate_small_cases_prompt


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


def enumerate_small_cases(problem_statement: str, original_size: str, small_sizes: list, focus_aspect: str) -> str:
    """
    Generate a prompt for exhaustively analyzing smaller versions of a combinatorics problem
    to identify patterns that lead to the solution of the full problem.
    
    Args:
        problem_statement: The full problem statement
        original_size: Description of the original problem size (e.g., "25 students", "9 balls")
        small_sizes: List of smaller sizes to analyze (e.g., [3, 4, 5, 6])
        focus_aspect: Specific aspect to focus on during enumeration
        
    Returns:
        A formatted prompt that guides systematic enumeration and pattern analysis
    """
    return format_enumerate_small_cases_prompt(
        problem_statement=problem_statement,
        original_size=original_size,
        small_sizes=small_sizes,
        focus_aspect=focus_aspect
    )


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


# Tool schema for the enumerate small cases tool
ENUMERATE_SMALL_CASES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "enumerate_small_cases",
        "description": "Systematically analyze smaller versions of a combinatorics problem to discover patterns. This tool helps identify scaling behavior, structural patterns, and the formula for larger cases by exhaustively examining small instances. Particularly useful for problems involving counting, optimization over discrete structures, or finding extremal configurations.",
        "parameters": {
            "type": "object",
            "properties": {
                "problem_statement": {
                    "type": "string",
                    "description": "The complete problem statement. Include all constraints, objectives, and any special conditions (e.g., 'arrangements equivalent under rotation are considered the same')."
                },
                "original_size": {
                    "type": "string",
                    "description": "A clear description of the size parameter in the original problem. Examples: '25 students', '9 balls on a circle', 'n=2025 max value', '100x100 grid'."
                },
                "small_sizes": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of small sizes to analyze exhaustively. Choose sizes that are small enough to enumerate completely but large enough to reveal patterns. Typically 3-7 values like [3,4,5,6] or [2,3,4,5,6]."
                },
                "focus_aspect": {
                    "type": "string",
                    "description": "Specific guidance on what to look for. Examples: 'Focus on worst-case configurations that maximize the objective', 'Look for all arrangements achieving the minimum, not just obvious ones', 'Pay attention to how values can be reused while satisfying constraints'."
                }
            },
            "required": ["problem_statement", "original_size", "small_sizes", "focus_aspect"]
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
    import traceback
    
    try:
        if tool_name == "submit_math_answer":
            # Validate required argument
            if "answer" not in arguments:
                raise ValueError("Missing required argument 'answer' for submit_math_answer tool")
            return submit_math_answer(arguments["answer"], ground_truth, verbose)
            
        elif tool_name == "enumerate_small_cases":
            # Validate required arguments
            required_args = ["problem_statement", "original_size", "small_sizes", "focus_aspect"]
            missing_args = [arg for arg in required_args if arg not in arguments]
            if missing_args:
                raise ValueError(f"Missing required arguments for enumerate_small_cases: {', '.join(missing_args)}")
            
            # Validate small_sizes is a list
            if not isinstance(arguments["small_sizes"], list):
                raise TypeError(f"Argument 'small_sizes' must be a list, got {type(arguments['small_sizes']).__name__}")
            
            # Validate all elements in small_sizes are integers
            if not all(isinstance(x, int) for x in arguments["small_sizes"]):
                raise TypeError("All elements in 'small_sizes' must be integers")
            
            return enumerate_small_cases(
                arguments["problem_statement"],
                arguments["original_size"],
                arguments["small_sizes"],
                arguments["focus_aspect"]
            )
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
            
    except Exception as e:
        # Create detailed error message with traceback
        error_lines = [
            f"❌ ERROR in {tool_name}: {type(e).__name__}: {str(e)}",
            "",
            "📍 Traceback:",
            traceback.format_exc(),
            "",
            "💡 Debugging hints:"
        ]
        
        # Add specific hints based on error type
        if isinstance(e, KeyError):
            error_lines.append(f"  - Check that you're passing all required arguments")
            error_lines.append(f"  - Received arguments: {list(arguments.keys())}")
        elif isinstance(e, TypeError):
            error_lines.append(f"  - Check that argument types match expected types")
            error_lines.append(f"  - See tool schema for correct types")
        elif isinstance(e, ValueError):
            error_lines.append(f"  - Check that argument values are valid")
            if "small_sizes" in str(e):
                error_lines.append(f"  - small_sizes should be a list of integers like [3, 4, 5]")
        
        error_msg = "\n".join(error_lines)
        
        if verbose:
            print(error_msg)
            
        return error_msg