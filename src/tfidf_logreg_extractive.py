
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
