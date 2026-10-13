from llm_sdk import Small_LLM_Model
from src.byte_decoder import decode, get_vocab, to_bytes
from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar
from pydantic import BaseModel, ConfigDict, Field
from src.models import TypeSpec
from src import Log

import json
import re
import sys
import termios


class LLM_constrained(BaseModel):
    """my wrapper, LLM Class with more methods

    holds the Small_LLM_Model, adds constrained generation on top of it
    let's cache the token IDs -> bytes
    special tokens are left out, partial UTF-8 bytes are kept
    """

    # Small_LLM_Model isn't a pydantic model: only checked with isinstance()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # loads the model when none is given
    llm: Small_LLM_Model = Field(default_factory=Small_LLM_Model)
    # filled in model_post_init, repr=False: it's ~150k entries
    token_bytes: dict[int, bytes] = Field(default_factory=dict, repr=False)

    # what the validators see in place of a char whose bytes aren't all
    # generated yet, eg b' \xf0\x9f' (half of ' 🚀') is shown as ' �'
    # it's not a digit, quote, comma, brace, space, backslash or control
    # char: numbers refuse it, strings accept it, it never closes a value
    PENDING: ClassVar[str] = "\ufffd"

    def model_post_init(self, context: Any, /) -> None:
        """Caches a token id -> bytes dict on init
        """
        vocab = get_vocab(self.llm)
        # special tokens aren't in vocab.json, they never get in
        self.token_bytes = {
            tok_id: to_bytes([tok_id], vocab) for tok_id in vocab}

    # :::::::::  :::::::::: ::::    ::::   ::::::::
    # :+:    :+: :+:        +:+:+: :+:+:+ :+:    :+:
    # +:+    +:+ +:+        +:+ +:+:+ +:+ +:+    +:+
    # +#+    +:+ +#++:++#   +#+  +:+  +#+ +#+    +:+
    # +#+    +#+ +#+        +#+       +#+ +#+    +#+
    # #+#    #+# #+#        #+#       #+# #+#    #+#
    # #########  ########## ###       ###  ########

    # Functions present for demonstration only

    def public_decode(self, ids: list[int]) -> str:
        """public reimplementation of .decode()"""
        return decode(ids, get_vocab(self.llm), True)

    def running_autocomplete(self) -> None:
        """
        run an autocomplete chat, just for testing purposes
        """

        def generate_no_thinking(prompt: str, max_tokens: int = 30) -> str:
            """
            greedily pick the answer to the prompt, prefill it so there's no
            thinking (Qwen is a thinking model) and so EOS appear in the scheme
            """
            prompt = "<|im_start|>user\n" + prompt + \
                "<|im_end|>\n<|im_start|>assistant\n" + \
                "<think>\n\n</think>\n\n" + prompt
            input_ids: list[int] = self.llm.encode(prompt)[0].tolist()
            # we dont want the LLM to decode "im_start", "user", etc..
            start_of_prompt = len(input_ids)
            eos = {
                self.llm.encode(t)[0].item() for t in (
                    "<|im_end|>",
                    "<|endoftext|>")}  # the two end tokens

            for _ in range(max_tokens):
                Log.progress()
                logits = self.llm.get_logits_from_input_ids(input_ids)
                next_id = max(range(len(logits)), key=logits.__getitem__)
                if next_id in eos:
                    break
                input_ids.append(next_id)

            return self.llm.decode(input_ids[start_of_prompt:])

        while True:
            # flushes previous input
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            prompt = input("(Enter a sentence, I will complete it)\n")
            if len(prompt) == 0:
                Log.error("Prompt cannot be empty")
                continue
            Log.info(generate_no_thinking(prompt))

    #  ::::::::  :::::::::: :::        :::::::::: :::::::: :::::::::::
    # :+:    :+: :+:        :+:        :+:       :+:    :+:    :+:
    # +:+        +:+        +:+        +:+       +:+           +:+
    # +#++:++#++ +#++:++#   +#+        +#++:++#  +#+           +#+
    #        +#+ +#+        +#+        +#+       +#+           +#+
    # #+#    #+# #+#        #+#        #+#       #+#    #+#    #+#
    #  ########  ########## ########## ########## ########     ###

    # the meat of the class, generate_from, generate_value and
    # generate_object discriminate by options or type,
    # they set up the params for _generate() to accept

    def generate_from(self, prompt: str,
                      options: list[str], end: str = '"') -> str:
        """Choose the best option given a prompt and a list of options.

        The model writes the option one token at a time, only tokens that
        keep at least one option possible are allowed. It stops as soon as
        a single option is left and return it.

        Args:
            prompt: the prompt so far, ending right before the option.
            options: the allowed answers. Duplicates are dropped.
            end: what must follow the option, eg '"' or ",". Lets the model
                pick an option that is a prefix of another, eg "a" vs "ab".

        Returns:
            The chosen option, without `end`.

        Raises:
            ValueError: if `options` is empty, or if the model can't
                produce a valid token.
        """

        def remaining(text: str) -> list[str]:
            """returns the options still compatible with text"""
            return [o for o in options if (o + end).startswith(text)]

        # drop duplicates, though it shouldnt be possible to have dups
        options = list(dict.fromkeys(options))
        if not options:
            raise ValueError("generate_from() needs at least one option")
        if len(options) == 1:  # nothing to choose, no model call
            return options[0]

        text = self._generate(
            prompt,
            is_valid=lambda text: len(remaining(text)) >= 1,  # still possible
            is_done=lambda text: len(remaining(text)) == 1,  # only one left
            # every token adds >= 1 byte, the text never outgrows option + end
            max_tokens=max(len((o + end).encode()) for o in options),
        )
        # because is_done was true, we know that remaining() = 1
        return remaining(text)[0]

    def generate_value(self, prompt: str, type_: TypeSpec,
                       end: str) -> float | int | bool | str:
        """Generate one value, constrained to a type.

        Allowed types: boolean, number, integer, string

        Args:
            prompt: the prompt so far, ending right before the value,
                eg '..."key":'
            type_: the expected type of the value.
            end: what must follow the value, eg "," or "}". Ignored for a
                string, its closing quote already ends it.

        Returns:
            The value as a Python value, converted by TypeSpec.convert()

        Raises:
            ValueError: if the model can't produce a valid value.
        """
        is_valid: Callable[[str], bool]
        match type_.type:
            case "boolean":
                return type_.convert(
                    self.generate_from(prompt, ["true", "false"], end))
            case "number":
                is_valid = partial(self.is_valid_number, end=end)
            case "integer":
                is_valid = partial(self.is_valid_number, end=end,
                                   integer=True)
            case "string":
                prompt += '"'
                # a string ends at its closing quote, even when the token
                # also carries what comes next, like ", or "}
                end = '"'
                is_valid = self.is_valid_string

        # a valid number only holds `end` at its very end
        text = self._generate(prompt, is_valid=is_valid,
                              is_done=lambda t: end in t)
        # keep what's before `end`: for strings, drops what followed the quote
        return type_.convert(text.partition(end)[0])

    def generate_object(self, prompt: str,
                        fields: dict[str, TypeSpec]) -> dict[str, Any]:
        """Generate a JSON object, one value per field, in the fields' order.

        We write the JSON around the values ({, "key":, commas), the
        model only writes the values. Each value is added to the prompt, so
        the next ones can see it.

        It's really just a wrapper / looper around .generate_value()

        Raises:
            ValueError: if the model can't produce a valid value (no
                allowed token, or it runs out of tokens).
        """
        # compact JSON, no spaces: after a lone ' ' token the model only
        prompt += "{"
        values: dict[str, Any] = {}
        last = len(fields) - 1
        for i, (key, type_) in enumerate(fields.items()):
            end = "}" if i == last else ","
            prompt += f'"{key}":'
            value = self.generate_value(prompt, type_, end)
            Log.info(f"{key} = {value!r}")
            values[key] = value
            # back to JSON text, quotes included for a string
            # ensure_ascii keeps non-ASCII chars instead of escaping them
            prompt += json.dumps(value, ensure_ascii=False) + end
        return values

    # generic value generator

    def _generate(self, prompt: str,
                  is_valid: Callable[[str], bool],
                  is_done: Callable[[str], bool], max_tokens: int = 42) -> str:
        """Generic constrained decoding, greedy.

        At each step:
            * Check if we're done.
            * Pick the token with the highest logit from allowed tokens.

        A token can hold half a char, eg b' \xf0\x9f' is half of ' 🚀'.
        Until its other bytes come, is_valid and is_done see PENDING in
        place of that char, at the end of the text.

        Args:
            prompt: the prompt the generated text continues.
            is_valid: takes the text generated so far plus a candidate
                token, says if it's allowed.
            is_done: takes the text generated so far, says if we're done.
            max_tokens: max number of tokens to generate.

        Returns:
            The generated text only, without the prompt.

        Raises:
            ValueError: if no token is allowed, or if max_tokens runs out
                before is_done is true.
        """

        def as_text(data: bytes) -> str:
            r"""Decode the generated bytes, a cut last char becomes PENDING.

            eg b'ab' -> 'ab', b'a \xf0\x9f' -> 'a \ufffd'

            Raises:
                UnicodeDecodeError: if the bytes can't be UTF-8, eg b'a\x80'
            """
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError as e:
                # only the last char is cut: the next tokens can complete it
                if e.reason == "unexpected end of data" and e.end == len(data):
                    return data[:e.start].decode("utf-8") + self.PENDING
                raise

        def is_allowed(data: bytes) -> bool:
            """is_valid() on bytes, those that can't be UTF-8 are refused"""
            try:
                return is_valid(as_text(data))
            except UnicodeDecodeError:
                return False

        input_ids: list[int] = self.llm.encode(prompt)[0].tolist()
        generated, text = b"", ""

        for _ in range(max_tokens):
            if is_done(text):
                break

            Log.progress()
            logits = self.llm.get_logits_from_input_ids(input_ids)

            # pieces are never empty: every candidate makes the output longer
            ids_allowed = [tok_id for tok_id, piece in self.token_bytes.items()
                           if is_allowed(generated + piece)]
            if not ids_allowed:
                raise ValueError(f"no valid token after {text!r}")

            # key= sort by logits[], will return id
            best_token = max(ids_allowed, key=logits.__getitem__)

            # keep the chosen token, update the output
            input_ids.append(best_token)
            generated += self.token_bytes[best_token]
            text = as_text(generated)

        Log.clear_line()
        # the loop can also stop because max_tokens ran out
        if not is_done(text):
            raise ValueError(
                f"ran out of tokens ({max_tokens}) after {text!r}")
        return text

    # ::::    ::::  ::::::::::: ::::::::   ::::::::
    # +:+:+: :+:+:+     :+:    :+:    :+: :+:    :+:
    # +:+ +:+:+ +:+     +:+    +:+        +:+
    # +#+  +:+  +#+     +#+    +#++:++#++ +#+
    # +#+       +#+     +#+           +#+ +#+
    # #+#       #+#     #+#    #+#    #+# #+#    #+#
    # ###       ### ########### ########   ########

    # various helpers

    @staticmethod
    def is_valid_number(text: str, end: str, integer: bool = False) -> bool:
        """True if `text` can still grow into a number, or is a complete
        number followed by `end`. With `integer`, no decimal part allowed.

        eg: end is ',': '', '-', '3', '3.', '-3.5' and '3.5,' are valid
        '.', '-.', '3.,', '3,,' and ' 3' are not, nor '3.' for an integer
        """
        if integer:
            complete, prefix = r"-?[0-9]+", r"-?[0-9]*"
        else:
            complete = r"-?[0-9]+(\.[0-9]+)?"
            prefix = r"-?([0-9]+(\.[0-9]*)?)?"
        if text.endswith(end):  # closed: what's before must be complete
            return re.fullmatch(complete, text[:-len(end)]) is not None
        # still open: any beginning of a number is fine
        return re.fullmatch(prefix, text) is not None

    @staticmethod
    def is_valid_string(text: str) -> bool:
        r"""Doesnt allow \ backslashes"""
        # the string closes at the first quotation mark
        # because , } ] or spaces may follow which is valid
        body, _, rest = text.partition('"')
        # c < ' ': control chars (newline, tab...)
        return (not any(c == '\\' or c < ' ' for c in body)
                and all(c in ',}]' or c.isspace() for c in rest))


if __name__ == "__main__":
    self = LLM_constrained()
    self.running_autocomplete()
