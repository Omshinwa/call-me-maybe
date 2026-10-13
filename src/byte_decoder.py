"""public implementation of llm.decode(ids)"""

from llm_sdk import Small_LLM_Model
from src import Log
import json

# TOKEN : a token in LLM represents a sequence of bytes (256)
# LOGIT : during prediction, a value given to each token
# (from negative inf to positive inf)
# raw, unnormalized score before turning them into probability

# WORD -> token(s) -> token(s) ids -> sequence of bytes

# "élément" is two tokens, token 28882 `Ã©lÃ©` + token 478 `ment`.
# élé : [195, 169, 108, 195, 169]
# ment: [109, 101, 110, 116]

from functools import lru_cache


@lru_cache(maxsize=1)
def get_char_to_256() -> dict[str, int]:
    """
    builds the table of char -> 256
    """
    visible = [*range(ord("!"), ord("~") + 1),  # * is the unpacking op
               *range(ord("¡"), ord("¬") + 1),
               *range(ord("®"), ord("ÿ") + 1)]
    table = {chr(b): b for b in visible}
    n = 0
    for b in range(256):
        if b not in visible:
            table[chr(256 + n)] = b
            n += 1
    return table


CHAR_TO_BYTE = get_char_to_256()


@lru_cache(maxsize=1)
def get_vocab(llm: Small_LLM_Model) -> dict[int, str]:
    """Given an llm, get its vocab as a dict"""
    vocab_path = llm.get_path_to_vocab_file()
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    return {v: k for k, v in vocab.items()}


def to_bytes(ids: list[int], vocab: dict[int, str]) -> bytes:
    """The raw bytes of the tokens, before any UTF-8 decoding.
    A token can hold half a char:
    ' 🚀' is 3 tokens: b' \\xf0\\x9f' + b'\\x9a' + b'\\x80'
    """
    byte_list: list[str] = []
    for i in ids:
        if i not in vocab:
            continue
        byte_list += [c for c in vocab[i]]
    return bytes(CHAR_TO_BYTE[c] for c in byte_list)


def decode(ids: list[int], vocab: dict[int, str], hide_errors: bool) -> str:
    """Return the human word. Given a list to token ids.
    Returns should be identical to llm.decode(ids)
    """
    # the errors (such as uncomplete bytes) are replaced with �
    # note: this is the .decode() method of bytes lol
    res = to_bytes(ids, vocab).decode("utf-8", errors="replace")
    if hide_errors:
        res = res.replace("\ufffd", "")
    return res


if __name__ == "__main__":
    # run this main to see an explicit example of how LLM stores tokens as
    # bytes
    # test = " 🎉"
    test = " 🚀 moon"

    llm = Small_LLM_Model()
    VOCAB = get_vocab(llm)

    Log.info(f"encoding `{test}`")
    # encode always wraps the result in one extra list
    # the batch size is always 1 and [0] is always safe.
    ids = llm.encode(test)[0].tolist()
    # using llm.encode() to test and get the list of int
    Log.info(f"token ids repr: {[i for i in ids]}")
    # char repr
    Log.info("".join(str(list(VOCAB[i])) for i in ids))
    # byte repr
    Log.info("".join(
        "[" + ", ".join(f"{CHAR_TO_BYTE[c]:3}" for c in VOCAB[i]) + "]"
        for i in ids))

    res = llm.decode(ids)
    Log.info(f"llm.decoder gives `{res}`")
    res2 = decode(ids, VOCAB, False)
    Log.info(f"my decoder gives `{res2}`")
