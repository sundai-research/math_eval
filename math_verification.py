"""
Math verification utilities extracted from simple_math_evaluator.py
"""
import re
from math_verify import parse, verify


def extract_boxed(text):
    """
    Extract content from the last occurrence of \\boxed{} with support for nested braces.
    Returns a list with the extracted content (or empty list if not found).
    Handles both single backslash (from JSON strings) and double backslash cases.
    """
    # First, check if we have a malformed string from single backslash in JSON
    # Common case: '\boxed{...}' becomes '\x08oxed{...}' due to \b escape sequence
    if '\x08oxed{' in text:
        # Fix the malformed string by replacing backspace+oxed with proper \boxed
        text = text.replace('\x08oxed{', '\\boxed{')
    
    # Also handle other potential escape sequences that might occur
    # \b -> backspace, \f -> form feed, \n -> newline, \r -> carriage return, \t -> tab
    for escape_char, replacement in [('\x08', '\\b'), ('\x0c', '\\f'), ('\n', '\\n'), ('\r', '\\r'), ('\t', '\\t')]:
        if escape_char + 'oxed{' in text:
            text = text.replace(escape_char + 'oxed{', '\\boxed{')
    
    stack = []
    boxed_contents = []
    i = 0
    start_idx = -1

    while i < len(text):
        if text[i : i + 7] == "\\boxed{" and (i == 0 or text[i - 1] != "\\"):
            if not stack:
                start_idx = i + 7
            stack.append("{")
            i += 7
        elif text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            stack.append("{")
            i += 1
        elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
            if stack:
                stack.pop()
                if not stack and start_idx != -1:
                    boxed_contents.append(text[start_idx:i])
                    start_idx = -1
            i += 1
        else:
            i += 1

    # Return the last boxed content if any were found
    if boxed_contents:
        return [boxed_contents[-1]]  # Return only the last match

    # Fallback to regex if the first method fails
    pattern = r"\\boxed{((?:[^{}]|{(?:[^{}]|{[^{}]*})*})*?)}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return [matches[-1].group(1)]  # Return the last match

    return []


def format_for_math_verify(answer):
    """
    Format the answer for math verification, ensuring it has $ symbols
    """
    if not answer:
        return "$.$"  # Return a default value to avoid empty strings

    # Remove any existing dollar signs and surrounding whitespace
    answer = answer.strip()

    # Clear dollar signs at beginning and end
    if answer.startswith("$"):
        answer = answer[1:]
    if answer.endswith("$"):
        answer = answer[:-1]

    # Remove remaining whitespace
    answer = answer.strip()

    # Ensure content is not empty
    if not answer:
        return "$.$"

    # Add dollar signs
    return f"${answer}$"


def string_compare_answers(extracted, gold):
    """
    Compare answers using string normalization as a fallback when math_verify fails
    """
    # Clean and normalize strings
    def normalize(text):
        if not text:
            return ""
        # Remove all whitespace
        text = re.sub(r"\s+", "", text)
        # Replace common equivalent representations
        text = text.replace("\\frac", "")
        text = text.replace("\\cdot", "*")
        text = text.replace("\\times", "*")
        # Remove all LaTeX commands
        text = re.sub(r"\\[a-zA-Z]+", "", text)
        return text

    normalized_extracted = normalize(extracted)
    normalized_gold = normalize(gold)

    # Direct comparison or check for inclusion
    return (
        normalized_extracted == normalized_gold
        or normalized_gold in normalized_extracted
        or normalized_extracted in normalized_gold
    )


def verify_answer(extracted: str, gold: str) -> bool:
    """
    Verify if extracted answer matches gold answer.
    First tries math_verify, then falls back to string comparison.
    """
    # Try math_verify first
    try:
        formatted_gold = format_for_math_verify(gold)
        formatted_extracted = format_for_math_verify(extracted)
        
        gold_parsed = parse(formatted_gold)
        extracted_parsed = parse(formatted_extracted)
        
        return verify(gold_parsed, extracted_parsed)
    except Exception:
        # Fallback to string comparison
        try:
            return string_compare_answers(extracted, gold)
        except Exception:
            # If even string comparison fails, return False
            return False
        
def verify_answer_with_math_verify(generated: str, gold: str) -> bool:
    """
    Verify if extracted answer matches gold answer.
    First tries math_verify, then falls back to string comparison.
    """
    extracted = extract_boxed(generated)[0]
    print(f"extracted: {extracted}")
    print(f"gold: {gold}")
    return verify_answer(extracted, gold)