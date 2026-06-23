# tests/test_vocabulary.py
from vocabulary import generate_workplace_vocabulary

def test_returns_nonempty_unique_strings():
    vocab = generate_workplace_vocabulary()
    assert isinstance(vocab, list)
    assert len(vocab) >= 20
    assert all(isinstance(v, str) and v for v in vocab)
    assert len(vocab) == len(set(vocab))  # no duplicates
