*This project has been created as part of the 42 curriculum by wiwu.*

# Call me maybe!

Function calling with a 0.6B model (Qwen3-0.6B), through constrained decoding.

A request in plain English goes in, a structured JSON file comes out.


```
"What is the sum of 2 and 3?"  ->  {"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

## Description

The program reads a list of available functions
(`functions_definition.json`) and a list of requests
(`function_calling_tests.json`). For each request it asks
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) which function to call
and with which arguments, then writes all the answers to one JSON file.

Small language models are unreliable at generating structured output. We don't trust them to produce JSON. Instead, at every step, we force the model to only pick a token that fits the schema and JSON structure (**constrained decoding**). The program then writes the JSON structure itself; the model only:

- picks the function name, among the names that exist;
- writes each argument value, with the correct type.

## Instructions

Requirements: Python 3.10 or later and [uv](https://docs.astral.sh/uv/). The
first run downloads the model weights from Hugging Face.

```sh
make install   # uv sync: creates .venv and installs the dependencies
make run       # uv run python -m src, with the default paths
make debug     # the same, under pdb
make lint      # flake8 . and mypy . with the subject's flags
make clean     # removes __pycache__, .mypy_cache, .pytest_cache
make fclean    # clean, and removes .venv
```

optional arguments:

| argument                 | default path                                |
|--------------------------|---------------------------------------------|
| `--functions_definition` | `data/input/functions_definition.json`      |
| `--input`                | `data/input/function_calling_tests.json`    |
| `--output`               | `data/output/function_calling_results.json` |

`make run ARGS="--input my_prompts.json"`. The output folder is created if it
doesn't exist.

Supported parameter types: `number`, `integer`, `string`, `boolean`.
Nested types (`object`, `array`) are not supported: a functions file
that uses them is rejected with an error message and exit code 1.

## Example usage

```sh
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

Output file:

```json
[
  {
    "prompt": "What is the sum of 265 and 345?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 265.0,
      "b": 345.0
    }
  },
  ...
]
```


## Algorithm explanation

### Caching the vocabulary

Qwen uses byte-level BPE: each token stands for a sequence of bytes. In
`vocab.json` those bytes are written as printable characters (GPT-2
style: a space becomes `Ġ`, a newline `Ċ` etc...).

We cache a `token id -> bytes` dict (151,643 entries), it's the model's vocabulary. We leave out special instruction tokens like `<|im_end|>`.

We take care in keeping tokens that are only part of a UTF-8 character:

eg: `" 🚀"` is three tokens, `b' \xf0\x9f'`, `b'\x9a'` and `b'\x80'`, and none of them is a full character on its own.

### Worker abstraction

When generating a value, we create a prompt that asks the LLM only what we need (we never want it to generate the full JSON). We give it all the necessary information (the available functions, the user prompt, ..). This is abstracted using our class `Worker`.

### The generic loop: `_generate`

Then, for each blank in our JSON structure, we build the text, token by token:

We pass the LLM the prompt, get back the logits of every token.
From that list, we only keep the ones that keep the schema correct.
Then we greedily pick the highest logit.

To keep the schema correct, the generic `_generate` function calls two functions taken as argument:

- `is_valid(text)`: can `text` still become a correct answer?
- `is_done(text)`: is `text` a complete answer?

We pass functions such as `is_valid_number` or `is_valid_string`.

`is_valid` is used to filter the tokens, and as soon as `is_done` is `True` (or if there's only 1 choice possible left) we return the value.

That generated text is converted to its type, stored in a list of `Answers`. This list is then what produces the JSON output.

## Design decisions

- **The model writes values, never structure.** Keys, their order, braces and
  commas come from `functions_definition.json`, so the schema can't be broken,
  and there are fewer tokens to generate.
- **A different prompt for function names.** Choosing a function doesn't need JSON at all: the model only answers "which of these?". The second prompt shows only the chosen function, so the model can't borrow a parameter from another one.
- **Greedy decoding.** The same input always gives the same output. Also faster.
- **We stop as early as possible.** As soon as only 1 remaining option is possible, we can just return it early and save computation.
- **Filtering the whole vocabulary, no token pool.** An alternative coding design for filtering tokens involved constructing a pool of candidate tokens built once before generating (instead, we filter the whole vocabulary at each step). This was especially interesting for function names, as we knew all the possibilities. This would have sped up the search. I chose not to because it only benefits function names, I wanted to keep `_generate` with `is_valid` being more generic and cleaner code.
- **Compact JSON, no spaces.** See *Negative numbers* in *Challenges faced*.
- **No `\` in strings.** See *Challenges faced*.
- **Byte-level decoding instead of ASCII only.** Accents, Chinese
  characters and emoji work in string arguments, even the ones split across several tokens. The price is a bigger vocabulary to scan and a UTF-8 decode per candidate, but it's way cooler to have emoji arguments.
- **A failing request is skipped, not written half-filled.** The output only contains complete, schema-valid answers; the failure is printed.
- **Preventing Qwen3 from thinking.** The model is trained to open its answer with a `<think>` block. Prefilling an empty `<think></think>` puts it straight into answering.
- **No numpy.** I used numpy arrays before to speed up the handling of vector of ints. Now I cache everything in a dict of token id -> bytes and only return a list of valid ints, which is faster.
- **No setting to -inf.** The subject asks us to set the logits to -infinity. I just remove the invalid tokens which is equivalent.

## Performance analysis

Measured on an Apple Silicon MacBook with 8 GB of RAM:

|                                              | result                     |
|----------------------------------------------|----------------------------|
| subject's 11 requests                        | 11/11 right function and arguments |
| extra test cases (11 requests)               | 11/11, see *Testing strategy* |
| valid JSON                                   | 100%, by construction      |
| full run on the subject's file, model loading included | 1 min 33 s       |

**Speed.** The 11 requests take 106 generated tokens. Each one costs:

1. a forward pass over the whole prompt: the SDK keeps no cache between
   calls, so the prompt is recomputed at every step;
2. a scan of the 151k cached tokens: each `generated + piece` is decoded
   from UTF-8, then tested with `is_valid`. It is about 60 ms in Python with
   the cache of token bytes.

**Memory.** The model needs free RAM. On the same 8 GB machine, while the
system was swapping (6 GB of swap in use), the same run took 14 minutes at
22% CPU.

## Challenges faced

### Prompting Qwen

I had to learn special tokens and how to prompt Qwen properly (using `<|im_start|>` or `<think>`) and inline the function parameters and information into the prompt.

### Names sharing a prefix & The end argument

With two functions sharing a prefix, eg: `fn_print` and `fn_print_formatted`, a naive solution would only pick the shorter name. That's why I require the model to output a closing text character.

The target text needs to be in the form of `some text + the character that comes right after in the JSON` (such as `"` or `,` or both `,"`). If we ask for a number for example, we keep generating until we hit a `,`. This way, we give the model a way to close the parameter.

### Half characters

`"🚀"` alone is one token, but `" 🚀"` (with the space) is three, `b' \xf0\x9f'`, `b'\x9a'` and `b'\x80'`. None of them a full character. Decoded alone, such a piece gives `�`. Early in development, I cached only the strings `token str -> token id` and dropped every `�`, so the model could only write characters that exist as whole tokens. It would never be able to write `'to the moon 🚀'` it would write `to the moon🚀`. Now every token is cached as bytes and the text is built as bytes, not as a string. Before `is_valid` is asked, each candidate `generated + piece` is decoded:

- not valid UTF-8 (eg a lone `\x80`): the token is refused;
- the last character is cut (eg `party time \xf0\x9f`): `is_valid` sees it as
  `�` (`PENDING`), so `party time �`. `�` is not a digit, quote, backslash or
  control char: numbers refuse it, strings accept it. The validators never
  deal with bytes;
- `is_done` is never asked while a character is cut.

### Escapes in strings

A previous version allowed the JSON escapes `\"` and
`\\`. It needed complex regex and code (finding the real closing quote, undoing the
escapes) and didn't help: the model never wrote `\"`, and once it wrote a
`\` it could loop. I decided to keep it simple and disallow escape chars.

### Negative numbers

After `"a": ` (with a space) the model almost never started a value with `-`, so negative numbers came out positive. Instead I prompt without whitespace.

## Testing strategy

Progressive anti regression testing strategy.

- `make lint`: flake8 and mypy with the subject's flags, no errors.
- `test/test_encoder.py`: my decoder gives the same text as the SDK's
  `decode` on accents, emoji and arbitrary ids; my encoder cuts the text
  around special tokens correctly.
- `test/test_json.py`, parsing: 7 bad function files must be rejected
  (duplicate names, unknown parameter type, unknown return type, missing
  field, empty list, not a list, malformed JSON), and the subject's files must
  load.
- `test/test_json.py`, prompting: each folder of `test/data/prompting/` holds
  `functions.json`, `prompts.json` and `expected.json`. Each request goes
  through the same `solve()` as the program and is compared to the expected
  answer. The files use the program's own input format, so a case can be run
  by hand:
  `make run ARGS="--functions_definition test/data/prompting/unicode/functions.json --input test/data/prompting/unicode/prompts.json"`
- By hand: missing file, invalid JSON and unsupported types give a message
  and exit code 1, with no traceback. Two full runs gave identical output.

`uv run pytest`: 21 passed. Two cases are there to show limits and
regressions:

- `escapes-0`: `Print 'C:\Users\me'` expects `C:/Users/me`. With `\`
  banned, the model swaps it for `/` (see *Escapes in strings*).
- `unicode-3`: `Print 'to the moon 🚀'` expects the space before `🚀`. It
  failed before the byte change (see *Half characters*).

## Resources

References:

- B. Willard, R. Louf, *Efficient Guided Generation for Large Language
  Models* (2023), the paper behind Outlines:
  <https://arxiv.org/abs/2307.09702>
- Qwen3-0.6B model card: <https://huggingface.co/Qwen/Qwen3-0.6B>
- Hugging Face LLM course, Byte-Pair Encoding tokenization:
  <https://huggingface.co/learn/llm-course/chapter6/5>
- pydantic `TypeAdapter`:
  <https://docs.pydantic.dev/latest/concepts/type_adapter/>

Other 42 implementations I compared mine to:

- <https://github.com/AlexHysel/42-Call-Me-Maybe>: interestingly hardcodes its regex
  mapping, reimplements the encoder with numpy.
- <https://github.com/ayfadli/Call-Me-Maybe>: short, caches with numpy, but
  doesn't handle UTF-8.
- <https://github.com/sousampere/42_call_me_maybe_v1.2> and its
  [review issue](https://github.com/sousampere/42_call_me_maybe_v1.2/issues/1):
  the model writes the whole JSON, there is no token limit, and a
  "starts with" check always picks the shortest of `ft_add` / `ft_add_two`.
  My version avoids all three.

### How AI was used

Claude (through Claude Code) was used to:

- help me understand how LLM models work
- write code following my specific instructions
- review the project against the subject (missing docstrings, Makefile rules,
  README requirements); write tests.

## Bonus

- Public reimplementation of `decode` (`src/byte_decoder.py`), tested against
  the SDK's.
- Public reimplementation of `encode` (`src/byte_encoder.py`): BPE with the correct merge order.
- Caching: the bytes of every token are computed once, at load time.
- A test suite: parsing errors and end-to-end cases.

# Improvement / Limitations

The greedy pick has some limitations, for example is most functions have a name starting with `fn`, but one is named `is_even_or_odd`, the model will still pick `fn` for the majority of the time. From there, `is_even_or_odd` would never be picked.

A fix is to have `generate_from`, score each whole option instead of choosing greedily. Add up the log-probabilities of each option's tokens (plus the closing ") and take the highest.