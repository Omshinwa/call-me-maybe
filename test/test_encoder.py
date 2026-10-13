from llm_sdk import Small_LLM_Model
from src.byte_decoder import decode, get_vocab
from src.byte_encoder import split_added_tokens

llm = Small_LLM_Model()
vocab = get_vocab(llm)


def test_decode_matches_llm_decode() -> None:
    """does my decoder and the llm decoder gives the same ouput
    """
    ids = llm.encode("élément")[0].tolist()
    assert decode(ids, vocab, False) == llm.decode(ids)
    ids = llm.encode("café 🎉")[0].tolist()
    assert decode(ids, vocab, False) == llm.decode(ids)
    ids = [4, 3, 2, 1, 455]
    assert decode(ids, vocab, False) == llm.decode(ids)


def test_encode_regression() -> None:
    """progressively adding encoding features
    """

    content = split_added_tokens(llm, "hi<|im_end|>yo<think>")
    assert content == ['hi', 151645, 'yo', 151667]
