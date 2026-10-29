# Vendored from Gorilla / Berkeley Function Calling Leaderboard (https://github.com/ShishirPatil/gorilla), Apache License 2.0;
# see VENDORING.md in this directory. Only this two-line attribution header was added; the body is the upstream copy.
"""
Standalone AST-based function call checker extracted from BFCL v3.

Source: https://github.com/ShishirPatil/gorilla
    berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py

Modifications from the original:
- Removed Java/JavaScript type handling (Python-only)
- Removed MODEL_CONFIG_MAPPING dependency (no model-specific function name conversion)
- Removed Language enum dependency
- Simplified function signatures accordingly

Original license: Apache 2.0
"""

import re


#### Constants ####

PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}

# Types that need recursive inner-type checking
PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]


#### Main entry point ####


def irrelevance_function_checker(model_output):
    """Check that model correctly abstained (produced no tool calls)."""
    if not model_output or model_output == [{}]:
        return {"valid": True, "error": []}
    return {
        "valid": False,
        "error": [f"Model should abstain but produced {len(model_output)} call(s)."],
        "error_type": "irrelevance_function_checker:should_abstain",
    }


def ast_checker(
    func_description,
    model_output,
    possible_answer,
    test_category: str = "simple",
):
    """Check function calls using AST-based comparison.

    Args:
        func_description: List of tool schema dicts, each with
            ``{"name": ..., "parameters": {"properties": ..., "required": [...]}}``.
        model_output: List of dicts ``[{func_name: {param: value}}, ...]``.
        possible_answer: List of dicts ``[{func_name: {param: [acceptable_values]}}, ...]``.
        test_category: One of ``"simple"``, ``"parallel"``, ``"multiple"``, ``"irrelevance"``.

    Returns:
        ``{"valid": bool, "error": list[str], "error_type": str}``
    """
    if "irrelevance" in test_category:
        return irrelevance_function_checker(model_output)
    if "parallel" in test_category:
        return parallel_function_checker_no_order(
            func_description, model_output, possible_answer
        )
    elif "multiple" in test_category:
        return multiple_function_checker(
            func_description, model_output, possible_answer
        )
    else:
        if len(model_output) != 1:
            return {
                "valid": False,
                "error": ["Wrong number of functions."],
                "error_type": "simple_function_checker:wrong_count",
            }
        return simple_function_checker(
            func_description[0], model_output[0], possible_answer[0]
        )


#### Helper utilities ####


def find_description(func_descriptions, name):
    if isinstance(func_descriptions, list):
        for fd in func_descriptions:
            if fd["name"] == name:
                return fd
        return None
    return func_descriptions


def get_possible_answer_type(possible_answer: list):
    for answer in possible_answer:
        if answer != "":  # empty string = optional parameter sentinel
            return type(answer)
    return None


def standardize_string(input_string: str):
    """Remove spaces and common punctuation, lower-case, normalise quotes.

    This avoids penalising trivial formatting differences such as
    ``"April 1, 2024"`` vs ``"April 1,2024"`` vs ``"April 1 2024"``.
    """
    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


#### Value checkers ####


def type_checker(
    param,
    value,
    possible_answer,
    expected_type_description,
    expected_type_converted,
    nested_type_converted,
):
    """Check that *value* has the correct type (one level of nesting)."""
    result = {
        "valid": True,
        "error": [],
        "is_variable": False,
        "error_type": "type_error:simple",
    }

    is_variable = False
    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None:
        if possible_answer_type != expected_type_converted:
            is_variable = True

    if type(value) == expected_type_converted:
        if nested_type_converted is None:
            result["is_variable"] = is_variable
            return result
        else:
            for possible_answer_item in possible_answer:
                flag = True
                if type(possible_answer_item) == list:
                    for value_item in value:
                        checker_result = type_checker(
                            param,
                            value_item,
                            possible_answer_item,
                            str(nested_type_converted),
                            nested_type_converted,
                            None,
                        )
                        if not checker_result["valid"]:
                            flag = False
                            break
                if flag:
                    return {"valid": True, "error": [], "is_variable": is_variable}

            result["valid"] = False
            result["error"] = [
                f"Nested type checking failed for parameter {repr(param)}. "
                f"Expected outer type {expected_type_description} with inner type "
                f"{str(nested_type_converted)}. Parameter value: {repr(value)}."
            ]
            result["error_type"] = "type_error:nested"

    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None:
        if type(value) == possible_answer_type:
            result["is_variable"] = True
            return result

    result["valid"] = False
    result["error"].append(
        f"Incorrect type for parameter {repr(param)}. Expected type "
        f"{expected_type_description}, got {type(value).__name__}. "
        f"Parameter value: {repr(value)}."
    )
    result["error_type"] = "type_error:simple"
    return result


def string_checker(param: str, model_output: str, possible_answer: list):
    standardized_possible = [
        standardize_string(a) for a in possible_answer if isinstance(a, str)
    ]
    standardized_output = standardize_string(model_output)

    if standardized_output not in standardized_possible:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {repr(param)}: {repr(model_output)}. "
                f"Expected one of {possible_answer}. Case insensitive."
            ],
            "error_type": "value_error:string",
        }
    return {"valid": True, "error": []}


def list_checker(param: str, model_output: list, possible_answer: list):
    standardized_output = list(model_output)
    for i in range(len(standardized_output)):
        if isinstance(standardized_output[i], str):
            standardized_output[i] = standardize_string(model_output[i])

    standardized_possible = []
    for i in range(len(possible_answer)):
        inner = []
        for j in range(len(possible_answer[i])):
            if isinstance(possible_answer[i][j], str):
                inner.append(standardize_string(possible_answer[i][j]))
            else:
                inner.append(possible_answer[i][j])
        standardized_possible.append(inner)

    if standardized_output not in standardized_possible:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {repr(param)}: {repr(model_output)}. "
                f"Expected one of {possible_answer}."
            ],
            "error_type": "value_error:list/tuple",
        }
    return {"valid": True, "error": []}


def dict_checker(param: str, model_output: dict, possible_answers: list):
    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}

    for i in range(len(possible_answers)):
        if possible_answers[i] == "":
            continue

        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        flag = True
        possible_answer = possible_answers[i]

        for key, value in model_output.items():
            if key not in possible_answer:
                result["valid"] = False
                result["error"].append(f"Unexpected dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

            standardize_value = value
            if isinstance(value, str):
                standardize_value = standardize_string(value)

            standardized_possible = []
            for j in range(len(possible_answer[key])):
                if isinstance(possible_answer[key][j], str):
                    standardized_possible.append(standardize_string(possible_answer[key][j]))
                else:
                    standardized_possible.append(possible_answer[key][j])

            if standardize_value not in standardized_possible:
                result["valid"] = False
                result["error"].append(
                    f"Invalid value for parameter {repr(key)}: {repr(value)}. "
                    f"Expected one of {standardized_possible}."
                )
                result["error_type"] = "value_error:dict_value"
                flag = False
                break

        for key, value in possible_answer.items():
            if key not in model_output and "" not in value:
                result["valid"] = False
                result["error"].append(f"Missing dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

        if flag:
            return {"valid": True, "error": []}

    return result


def list_dict_checker(param: str, model_output: list, possible_answers: list):
    result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}

    for answer_index in range(len(possible_answers)):
        flag = True

        if len(model_output) != len(possible_answers[answer_index]):
            result["valid"] = False
            result["error"] = ["Wrong number of dictionaries in the list."]
            result["error_type"] = "value_error:list_dict_count"
            flag = False
            continue

        for dict_index in range(len(model_output)):
            result = dict_checker(
                param,
                model_output[dict_index],
                [possible_answers[answer_index][dict_index]],
            )
            if not result["valid"]:
                flag = False
                break

        if flag:
            return {"valid": True, "error": []}

    return result


#### Function-level checkers ####


def simple_function_checker(
    func_description: dict,
    model_output: dict,
    possible_answer: dict,
):
    """Check a single function call against the expected answer.

    Args:
        func_description: Tool schema ``{"name": ..., "parameters": {"properties": ..., "required": [...]}}``.
        model_output: ``{func_name: {param: value}}``.
        possible_answer: ``{func_name: {param: [acceptable_values]}}``.
    """
    possible_answer = list(possible_answer.values())[0]
    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"].get("required", [])

    result = {
        "valid": True,
        "error": [],
        "error_type": "simple_function_checker:unclear",
    }

    # Check function name
    if func_name not in model_output:
        result["valid"] = False
        result["error"].append(
            f"Function name {repr(func_name)} not found in model output."
        )
        result["error_type"] = "simple_function_checker:wrong_func_name"
        return result

    model_params = model_output[func_name]

    # Check required parameters
    for param in required_params:
        if param not in model_params:
            result["valid"] = False
            result["error"].append(f"Missing required parameter: {repr(param)}.")
            result["error_type"] = "simple_function_checker:missing_required"
            return result

    # Validate types and values
    for param, value in model_params.items():
        if param not in param_details or param not in possible_answer:
            result["valid"] = False
            result["error"].append(f"Unexpected parameter: {repr(param)}.")
            result["error_type"] = "simple_function_checker:unexpected_param"
            return result

        full_param_details = param_details[param]
        expected_type_description = full_param_details.get("type", "string")
        nested_type_converted = None

        expected_type_converted = PYTHON_TYPE_MAPPING.get(expected_type_description, str)
        if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
            nested_type = full_param_details.get("items", {}).get("type", "string")
            nested_type_converted = PYTHON_TYPE_MAPPING.get(nested_type, str)

        # tuple → list (JSON round-trip loses tuple info)
        if expected_type_description == "tuple" and isinstance(value, tuple):
            value = list(value)

        # int → float auto-conversion
        if expected_type_description == "float" and isinstance(value, int):
            value = float(value)

        # Type checking
        type_check_result = type_checker(
            param,
            value,
            possible_answer[param],
            expected_type_description,
            expected_type_converted,
            nested_type_converted,
        )
        is_variable = type_check_result["is_variable"]
        if not type_check_result["valid"]:
            return type_check_result

        if not is_variable:
            if expected_type_converted == dict:
                result = dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

            elif expected_type_converted == list and nested_type_converted == dict:
                result = list_dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

            elif expected_type_converted == str:
                result = string_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

            elif expected_type_converted == list:
                result = list_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

        # Fallback: direct value check
        if value not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(
                f"Invalid value for parameter {repr(param)}: {repr(value)}. "
                f"Expected one of {possible_answer[param]}."
            )
            result["error_type"] = "value_error:others"
            return result

    # Check that all non-optional params are present
    for param in possible_answer:
        if param not in model_params and "" not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(
                f"Optional parameter {repr(param)} not provided and not marked as optional."
            )
            result["error_type"] = "simple_function_checker:missing_optional"
            return result

    return result


def parallel_function_checker_no_order(
    func_descriptions: list,
    model_output: list,
    possible_answers: list,
):
    """Check parallel (unordered) function calls."""
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "parallel_function_checker_no_order:wrong_count",
        }

    matched_indices = []

    for i in range(len(possible_answers)):
        func_name_expected = list(possible_answers[i].keys())[0]
        func_description = find_description(func_descriptions, func_name_expected)

        all_errors = []
        result = None

        for index in range(len(model_output)):
            if index in matched_indices:
                continue

            result = simple_function_checker(
                func_description,
                model_output[index],
                possible_answers[i],
            )

            if result["valid"]:
                matched_indices.append(index)
                break
            else:
                all_errors.append(
                    {
                        f"Model Result Index {index}": {
                            "sub_error": result["error"],
                            "sub_error_type": result["error_type"],
                            "model_output_item": model_output[index],
                            "possible_answer_item": possible_answers[i],
                        }
                    }
                )

        if result is None or not result["valid"]:
            considered_indices = [
                j for j in range(len(model_output)) if j not in matched_indices
            ]
            all_errors.insert(
                0,
                f"Could not find a matching function among index "
                f"{considered_indices} of model output for index {i} of possible answers.",
            )
            return {
                "valid": False,
                "error": all_errors,
                "error_type": "parallel_function_checker_no_order:cannot_find_match",
            }

    return {"valid": True, "error": []}


def multiple_function_checker(
    func_descriptions: list,
    model_output: list,
    possible_answers: list,
):
    """Check multiple sequential function calls (same function, different args)."""
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "multiple_function_checker:wrong_count",
        }

    func_name_expected = list(possible_answers[0].keys())[0]
    func_description = find_description(func_descriptions, func_name_expected)
    return simple_function_checker(
        func_description,
        model_output[0],
        possible_answers[0],
    )
