import argparse

from pydantic import TypeAdapter, ValidationError

from src.models import FunctionDef, Prompt

DEFAULT_FUNC_PATH = "data/input/functions_definition.json"


def get_function_def(path: str = DEFAULT_FUNC_PATH) -> list[FunctionDef]:
    """Load and check the function def file
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            functions = TypeAdapter(
                list[FunctionDef]).validate_json(
                file.read())
    except (OSError, ValidationError) as e:
        raise Exception(e)
    if not functions:
        raise Exception("No function found.")
    seen: set[str] = set()
    for f in functions:
        if f.name in seen:
            raise Exception(f"Duplicate function name: {f.name}")
        seen.add(f.name)
    return functions


DEFAULT_PROMPT_PATH = "data/input/function_calling_tests.json"


def get_prompt(
        path: str = DEFAULT_PROMPT_PATH) -> list[Prompt]:
    """Load and check the prompts file
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            prompts = TypeAdapter(list[Prompt]).validate_json(file.read())
    except (OSError, ValidationError) as e:
        raise Exception(e) from e
    if not prompts:
        raise Exception("No prompt found.")
    return prompts


def parse_args() -> argparse.Namespace:
    """
    Are your arguments even in the correct format?
    """
    parser = argparse.ArgumentParser(
        description="Call me maybe: function calling with a small LLM")
    parser.add_argument("--functions_definition",
                        default=DEFAULT_FUNC_PATH,
                        help="JSON file describing the available functions")
    parser.add_argument("--input",
                        default=DEFAULT_PROMPT_PATH,
                        help="JSON file with the prompts to process")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json",
                        help="where to write the results")
    return parser.parse_args()


if __name__ == "__main__":
    get_function_def()
    get_prompt()
