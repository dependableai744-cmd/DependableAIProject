
import re
import string
from collections import Counter


def _normalize(text: str) -> str:
    """Lowercase, remove punctuation + articles, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> int:
    """Return 1 if normalised strings match exactly, else 0."""
    if not gold:
        return 0
    return int(_normalize(prediction) == _normalize(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 score between prediction and gold answer."""
    if not gold:
        return 0.0
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    common      = Counter(pred_tokens) & Counter(gold_tokens)
    num_same    = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
