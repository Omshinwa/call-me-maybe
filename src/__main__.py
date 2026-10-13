# uv run python -m src [--functions_definition <function_definition_file>]
# [--input <input_file>] [--output <output_file>]

# uv run python -m src
# --functions_definition data/input/functions_definition.json
# --input data/input/function_calling_tests.json
# --output data/output/function_calls.json
import sys
from src.LLM_constrained import LLM_constrained
from src.models import Worker
from src import Log

from src.parser import get_function_def, get_prompt, parse_args


if __name__ == "__main__":
    args = parse_args()
    Log.info(f"functions: {args.functions_definition}, "
             f"input: {args.input}, output: {args.output}")

    try:
        functions = get_function_def(args.functions_definition)
        prompts = get_prompt(args.input)
    except Exception as e:
        Log.error(str(e))
        sys.exit(1)

    # Loading model
    try:
        llm = LLM_constrained()
    except Exception as e:
        Log.error(f"Error when loading the model. Details: {e}")
        sys.exit(1)

    worker = Worker(jobs=prompts, functions=functions)
    worker.run(llm)

    try:
        worker.export_to_json(args.output)
    except OSError as e:
        Log.error(str(e))
        sys.exit(1)
