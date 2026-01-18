from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


def _clean_sent(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def split_sentences(text: str, min_chars: int = 20) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [_clean_sent(s) for s in sents if _clean_sent(s)]
    return [s for s in sents if len(s) >= min_chars]


@dataclass
class TfidfLogRegExtractive:
    """
    TF-IDF + Logistic Regression sentence scoring model (extractive summarization).
    Inference-ready: you can load a fitted model OR fit it on your side (privately).
    """
    vectorizer: TfidfVectorizer
    clf: LogisticRegression

    @staticmethod
    def create_default() -> "TfidfLogRegExtractive":
        vec = TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
        )
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        )
        return TfidfLogRegExtractive(vec, clf)

    def fit(self, sentences: List[str], labels: List[int]) -> "TfidfLogRegExtractive":
        X = self.vectorizer.fit_transform(sentences)
        self.clf.fit(X, np.array(labels, dtype=int))
        return self

    def score_sentences(self, sentences: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(sentences)
        # probability of class 1 (relevant)
        return self.clf.predict_proba(X)[:, 1]

    def summarize(
        self,
        article: str,
        max_sents: int = 2,
        redundancy_thr: float = 0.7,
    ) -> str:
        """
        Extractive summary by scoring sentences and selecting top ones,
        with redundancy control (cosine similarity in TF-IDF space).
        """
        sents = split_sentences(article)
        if not sents:
            return ""

        X = self.vectorizer.transform(sents)  # sparse
        scores = self.clf.predict_proba(X)[:, 1]
        order = np.argsort(-scores)

        selected: List[int] = []
        selected_vecs = None

        for idx in order:
            if not selected:
                selected.append(int(idx))
                selected_vecs = X[idx]
            else:
                # redundancy check: max cosine similarity to already selected
                sims = cosine_similarity(X[idx], selected_vecs).ravel()
                if float(np.max(sims)) < float(redundancy_thr):
                    selected.append(int(idx))
                    selected_vecs = X[selected].copy()

            if len(selected) >= max_sents:
                break

        selected = sorted(selected)
        return " ".join(sents[i] for i in selected).strip()


def toy_fit_demo_model() -> TfidfLogRegExtractive:
    """
    Tiny demo fit with hand-made labels.
    """
    examples = [
        ("The government announced new measures today. The decision was debated in parliament. A cat slept on the desk.", [1, 1, 0]),
        ("Researchers improved an NLP model. They evaluated strong baselines. The cafeteria menu changed.", [1, 1, 0]),
        ("The company released quarterly results. Revenue increased year over year. Someone mentioned the weather.", [1, 1, 0]),
    ]
    sentences = []
    labels = []
    for art, lab in examples:
        sents = split_sentences(art, min_chars=10)
        # align labels to the kept sentences (simple demo assumption)
        # if filtering changes count, just skip the example
        if len(sents) != len(lab):
            continue
        sentences.extend(sents)
        labels.extend(lab)

    model = TfidfLogRegExtractive.create_default()
    return model.fit(sentences, labels)
