import re

def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def lead_2(article_text: str) -> str:
    """
    Lead-2 baseline: returns the first two sentences of the article.
    """
    sents = split_sentences(article_text)
    return " ".join(sents[:2])

