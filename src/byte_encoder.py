from llm_sdk import Small_LLM_Model
from src.byte_decoder import CHAR_TO_BYTE, get_vocab
from functools import lru_cache
import json
import unicodedata
import regex

from src import Log


def encode(llm: Small_LLM_Model, text: str) -> list[int]:
    """Public reimplementation of the llm.encode()
    it takes words and spits the tokens list(int) like the LLM would.

    1. handle added special tokens, eg:
    151643 = <|endoftext|>
    151644 = <|im_start|>
    ...

    2. Normalize with NFC
    because `é` can be written in different ways

    3. Pre-tokenize with regex
    given by tokenizer.json, we precut the words

    "Hello world, it's 2026! \n\nfoo bar"
    ->
    ['Hello', ' world', ',', ' it', "'s", ' '
    '2', '0', '2', '6', '!'
    '  \n\n', 'foo', '  ', ' bar']

    4. Rewrite each utf-8 byte of the chunk as its printable char,
        the way vocab.json writes them.
        " world" -> "Ġworld"
        "é"      -> "Ã©"  (2 bytes: 195, 169)

    5. Apply the BPE merges
        merges.txt gives the order the LLM merged bytes pair together.
        We reproduce that.
    """
    # 1. handle special tokens
    presplit: list[str | int] = split_added_tokens(llm, text)

    output: list[str | int] = []

    for element in presplit:
        if isinstance(element, int):  # already a token id
            output.append(element)
            continue
        # 2. normalize
        normalized: str = unicodedata.normalize("NFC", element)
        # 3. cut the text into chunks using regex
        output.extend(PRETOKENIZE_PATTERN.findall(normalized))

    Log.debug("* * Step 3: presplit by regex:")
    Log.debug(f"{[o for o in output]}")

    # 4. convert each str to their GPT_2 repr
    for i, txt in enumerate(output):
        if isinstance(txt, str):
            output[i] = "".join(BYTE_TO_CHAR[b] for b in txt.encode("utf-8"))

    Log.debug("* * Step 4: GPT 2 byte repr:")
    Log.debug(f"{[o for o in output]}")

    token_to_id = get_token_to_id(llm)

    Log.debug("* * Step 5: Grouped pairs:")
    # 5. Byte Pair Encoding
    real_output: list[int] = []
    for i, txt in enumerate(output):
        if isinstance(txt, int):
            Log.debug(f"['{txt}']")
            real_output.append(txt)
            continue
        # convert to int
        chunk_str: list[str] = byte_pair_encoding(llm, txt)

        Log.debug(f"{[o for o in chunk_str]}")

        real_output.extend([token_to_id[chunk] for chunk in chunk_str])

    return real_output


@lru_cache(maxsize=1)
def get_added_tokens(llm: Small_LLM_Model) -> dict[str, int]:
    """Given an llm, get its added tokens as a dict content -> id
    e.g. {"<|im_end|>": 151645, ...}
    These are matched literally in the text and skip BPE.
    """
    tokenizer_path = llm.get_path_to_tokenizer_file()
    with open(tokenizer_path, encoding="utf-8") as f:
        tokenizer = json.load(f)
    return {t["content"]: t["id"] for t in tokenizer["added_tokens"]}


def split_added_tokens(
        llm: Small_LLM_Model, text: str) -> list[str | int]:
    """Step 1: cut the text around the added tokens.
    Added tokens become their id, the text between them stays a str
    "a<|im_end|>b" -> ["a", 151645, "b"]
    """

    added_tokens = get_added_tokens(llm)
    # try longest tokens first
    by_length = sorted(added_tokens, key=len, reverse=True)

    pieces: list[str | int] = []

    pattern = "|".join(regex.escape(t) for t in by_length)
    # pattern is
    # <\|object_ref_start\|>|<\|object_ref_end\|>|<\|vision_start\|>....
    pos = 0
    for match in regex.finditer(pattern, text):
        if match.start() > pos:
            # add the text between the start and the token
            pieces.append(text[pos:match.start()])
        pieces.append(added_tokens[match.group()])
        pos = match.end()

    # # non regex version:
    # match_search = 0 # match_search is the lower level pos
    # while match_search < len(text):
    #     for token in by_length:
    #         if text.startswith(token, match_search):
    #             if match_search > pos:
    #                 pieces.append(text[pos:match_search])
    #             pieces.append(added_tokens[token])
    #             match_search += len(token)
    #             pos = match_search
    #             break
    #     else:  # no token starts here: this char is plain text
    #         match_search += 1

    if pos < len(text):
        pieces.append(text[pos:])  # add remaining non token text
    return pieces


# pre_tokenizer Split regex from tokenizer.json
# needs the `regex` package: stdlib `re` has no \p{L} / \p{N}
PRETOKENIZE_PATTERN = regex.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


# inverse of CHAR_TO_BYTE: byte -> its printable char, e.g. 32 (" ") -> "Ġ"
BYTE_TO_CHAR = {b: c for c, b in CHAR_TO_BYTE.items()}


def byte_pair_encoding(llm: Small_LLM_Model, text: str) -> list[str]:
    """Takes one chunk that's already in the byte-level form
    then cut it to tokens with the correct merge order

    "Ġanteater" -> "Ġante" / "ater"
    """
    ranks = get_merge_ranks(llm)

    # text = "Ġanteater"
    symbols = list(text)
    # symbols = ['Ġ','a','n','t','e','a', ..]
    # Merge pairs until no adjacent pair has a rank.
    while len(symbols) > 1:
        # find the adjacent pair with the lowest rank
        pairs = [(symbols[i], symbols[i + 1])
                 for i in range(len(symbols) - 1)]
        # pairs = [(Ġ,a), (a,n), (n,t), ..]
        best = min(pairs, key=lambda p: ranks.get(p, float("inf")))
        if best not in ranks:  # no adjacent pair can merge anymore
            break
        # merge every occurrence of best, left to right
        merged: list[str] = []
        i = 0
        while i < len(symbols):
            if (i < len(symbols) - 1
                    and (symbols[i], symbols[i + 1]) == best):
                merged.append(symbols[i] + symbols[i + 1])
                i += 2  # skip the right half, it's consumed
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged  # ['Ġante', 'ater']
    return symbols


@lru_cache(maxsize=1)
def get_merge_ranks(llm: Small_LLM_Model) -> dict[tuple[str, str], int]:
    """merges.txt -> {("Ġ", "Ġ"): 0, ("ĠĠ", "ĠĠ"): 1, ("i", "n"): 2, ...}
    line order is the priority: lowest rank is merged first
    """
    ranks: dict[tuple[str, str], int] = {}
    with open(llm.get_path_to_merges_file(), encoding="utf-8") as f:
        next(f)  # skip "#version: 0.2"
        for rank, line in enumerate(f):
            a, b = line.split()
            ranks[(a, b)] = rank
    return ranks


@lru_cache(maxsize=1)
def get_token_to_id(llm: Small_LLM_Model) -> dict[str, int]:
    """inverse of get_vocab: {"Ġworld": ..., ...}"""
    return {s: i for i, s in get_vocab(llm).items()}


if __name__ == "__main__":
    test = "Hello world, it's 2026! \n\nfoo bar"
    test = "<|im_start|> 🚀 moon élément"

    llm = Small_LLM_Model()
    VOCAB = get_vocab(llm)

    Log.info(f"encoding `{test}`")
    Log.level = 0

    Log.info(f"my decoder gives `{encode(llm, test)}`")
    Log.info(f"llm.decoder gives `{llm.encode(test)}`")
