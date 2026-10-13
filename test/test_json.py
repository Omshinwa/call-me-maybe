import json
from pathlib import Path
from typing import Any

import pytest

from src.LLM_constrained import LLM_constrained
from src.models import Worker
from src.parser import get_function_def, get_prompt

DATA = Path(__file__).parent / "data"

# parsing:
# every file in data/parsing/ is a bad functions_definition file:
# duplicate fn names, unknown types, missing field, empty, broken json...


@pytest.mark.parametrize(
    "path", sorted((DATA / "parsing").glob("*.json")), ids=lambda p: p.stem)
def test_bad_function_def_is_rejected(path: Path) -> None:
    """get_function_def must refuse it"""
    with pytest.raises(Exception):
        get_function_def(str(path))


def test_default_files_are_accepted() -> None:
    """the subject's files must still load"""
    assert get_function_def()
    assert get_prompt()


# correct prompting:
# each folder in data/prompting/ has
#   functions.json  the functions available
#   prompts.json    the prompts to run
#   expected.json   the answers we want, same order as prompts.json
# same format as the program, so a case can also be run by hand with:
# make run ARGS="--functions_definition <case>/functions.json \
#                --input <case>/prompts.json"
#
# prefix_names: fn_add vs fn_add_contact, shouldnt always pick the shorter one
# unicode: french accents, chinese chars and emojis in a string argument
# escapes: \ backslash in a string argument, not allowed: becomes /
# negative_numbers: negative number and integer arguments


def load_cases() -> list[Any]:
    """one test per prompt, eg 'unicode-0', 'unicode-1'..."""
    cases = []
    for case in sorted(p for p in (DATA / "prompting").iterdir()
                       if p.is_dir()):
        with open(case / "expected.json", encoding="utf-8") as f:
            for i, want in enumerate(json.load(f)):
                cases.append(pytest.param(case, i, want,
                                          id=f"{case.name}-{i}"))
    return cases


@pytest.fixture(scope="module")
def llm() -> LLM_constrained:
    """loading the model is slow: once for all the prompting tests"""
    return LLM_constrained()


@pytest.mark.parametrize("case, i, want", load_cases())
def test_prompting(llm: LLM_constrained, case: Path, i: int,
                   want: dict[str, Any]) -> None:
    """run the i-th prompt of the case, compare with expected.json"""
    functions = get_function_def(str(case / "functions.json"))
    prompts = get_prompt(str(case / "prompts.json"))
    assert prompts[i].prompt == want["prompt"], "expected.json out of order"

    worker = Worker(jobs=prompts, functions=functions)
    assert worker.solve(llm, prompts[i]).model_dump() == want
