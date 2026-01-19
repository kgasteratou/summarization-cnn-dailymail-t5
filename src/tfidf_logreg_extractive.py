from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str, min_chars: int = 20) -> List[str]:
    """
    Minimal sentence splitter with basic filtering.
    """
    text = _clean(text)
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [_clean(s) for s in sents if _clean(s)]
    return [s for s in sents if len(s) >= min_chars]


@dataclass
class TfidfLogRegExtractive:
    """
    TF-IDF + Logistic Regression sentence scoring for extractive summarization.
    """
    vectorizer: TfidfVectorizer
    clf: LogisticRegression

    @staticmethod
    def create_default() -> "TfidfLogRegExtractive":
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
        )
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        )
        return TfidfLogRegExtractive(vec, clf)

    def fit(self, sentences: List[str], labels: List[int]) -> "TfidfLogRegExtractive":
        X = self.vectorizer.fit_transform(sentences)
        y = np.asarray(labels, dtype=int)
        self.clf.fit(X, y)
        return self

    def summarize(self, article: str, max_sents: int = 2, redundancy_thr: float = 0.7) -> str:
        """
        Scores sentences with LogReg and selects top sentences with redundancy control
        using cosine similarity in TF-IDF space.
        """
        sents = split_sentences(article)
        if not sents:
            return ""

        X = self.vectorizer.transform(sents)
        scores = self.clf.predict_proba(X)[:, 1]
        order = np.argsort(-scores)

        selected: List[int] = []
        for idx in order:
            if not selected:
                selected.append(int(idx))
            else:
                sims = cosine_similarity(X[idx], X[selected]).ravel()
                if float(np.max(sims)) < float(redundancy_thr):
                    selected.append(int(idx))

            if len(selected) >= max_sents:
                break

        selected = sorted(selected)
        return " ".join(sents[i] for i in selected).strip()


def toy_fit_demo_model() -> TfidfLogRegExtractive:
    """
    Tiny demo fit with hand-made labels (public-safe).
    This is NOT meant to be a real training setup; it only demonstrates the pipeline.
    """
    examples = [
        (
            "The government announced new measures today. The decision was debated in parliament. A cat slept on the desk.",
            [1, 1, 0],
        ),
        (
            "Researchers improved an NLP model. They evaluated strong baselines. The cafeteria menu changed.",
            [1, 1, 0],
        ),
        (
            "The company released quarterly results. Revenue increased year over year. Someone mentioned the weather.",
            [1, 1, 0],
        ),
    ]

    sentences: List[str] = []
    labels: List[int] = []

    for article, lab in examples:
        sents = split_sentences(article, min_chars=10)
        if len(sents) != len(lab):
            continue
        sentences.extend(sents)
        labels.extend(lab)

    model = TfidfLogRegExtractive.create_default()
    return model.fit(sentences, labels)
