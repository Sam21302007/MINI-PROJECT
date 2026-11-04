import re

SIGNAL_WEIGHTS = {
    "text": 0.4,
    "duplicate": 0.2,
    "unverified": 0.1,
    "new_account": 0.1,
    "burst": 0.1,
    "rating_mismatch": 0.1,
}

POSITIVE_WORDS = {
    "amazing", "best", "excellent", "fantastic", "great", "love", "perfect",
    "recommend", "wonderful",
}
NEGATIVE_WORDS = {
    "awful", "bad", "broken", "defective", "horrible", "poor", "terrible",
    "waste", "worst",
}


def _normalized_text(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def explain_text(review, model, vectorizer, limit=5):
    """Return words with strongest model contribution toward suspicious or genuine."""
    if not hasattr(model, "coef_") or not hasattr(vectorizer, "get_feature_names_out"):
        return {"fake_words": [], "genuine_words": []}

    classes = list(model.classes_)
    if 1 not in classes:
        return {"fake_words": [], "genuine_words": []}

    class_index = classes.index(1)
    if len(classes) != 2 or model.coef_.shape[0] != 1:
        return {"fake_words": [], "genuine_words": []}

    features = vectorizer.transform([str(review)])
    if features.nnz == 0:
        return {"fake_words": [], "genuine_words": []}

    weights = model.coef_[0] if class_index == 1 else -model.coef_[0]
    feature_names = vectorizer.get_feature_names_out()
    word_scores = {}

    for feature_index, feature_value in zip(features.indices, features.data):
        term = feature_names[feature_index]
        contribution = float(feature_value * weights[feature_index])
        words = term.split()
        if not words:
            continue
        per_word_contribution = contribution / len(words)
        for word in words:
            word_scores[word] = word_scores.get(word, 0.0) + per_word_contribution

    fake_words = sorted(
        ((word, score) for word, score in word_scores.items() if score > 0),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    genuine_words = sorted(
        ((word, score) for word, score in word_scores.items() if score < 0),
        key=lambda item: item[1],
    )[:limit]
    return {
        "fake_words": [word for word, _ in fake_words],
        "genuine_words": [word for word, _ in genuine_words],
    }


def score_review(review, model, vectorizer, history=None, metadata=None):
    """Return a triage score and the evidence available for one review."""
    metadata = metadata or {}
    history = history or []
    model_score = float(
        model.predict_proba(vectorizer.transform([str(review)]))[
            0, list(model.classes_).index(1)
        ]
    )
    components = {"text": (SIGNAL_WEIGHTS["text"], model_score)}
    evidence = []

    if model_score >= 0.6:
        evidence.append("Text model flagged language patterns")

    normalized = _normalized_text(review)
    if normalized and any(
        _normalized_text(previous.get("text", "")) == normalized
        for previous in history
    ):
        components["duplicate"] = (SIGNAL_WEIGHTS["duplicate"], 1.0)
        evidence.append("Exact duplicate text found in recent reviews")
    else:
        components["duplicate"] = (SIGNAL_WEIGHTS["duplicate"], 0.0)

    if "verified_purchase" in metadata and metadata["verified_purchase"] is not None:
        is_unverified = not bool(metadata["verified_purchase"])
        components["unverified"] = (SIGNAL_WEIGHTS["unverified"], float(is_unverified))
        if is_unverified:
            evidence.append("Purchase is not marked as verified")

    account_age = metadata.get("account_age_days")
    if account_age is not None:
        is_new_account = float(account_age) <= 14
        components["new_account"] = (SIGNAL_WEIGHTS["new_account"], float(is_new_account))
        if is_new_account:
            evidence.append(f"New reviewer account ({int(account_age)} days old)")

    reviews_last_hour = metadata.get("reviews_last_hour")
    if reviews_last_hour is not None:
        is_burst = float(reviews_last_hour) >= 5
        components["burst"] = (SIGNAL_WEIGHTS["burst"], float(is_burst))
        if is_burst:
            evidence.append(f"High review volume ({int(reviews_last_hour)} in the last hour)")

    rating = metadata.get("rating")
    if rating is not None:
        words = set(_normalized_text(review).split())
        positive = bool(words & POSITIVE_WORDS)
        negative = bool(words & NEGATIVE_WORDS)
        mismatch = (float(rating) >= 4 and negative) or (float(rating) <= 2 and positive)
        components["rating_mismatch"] = (
            SIGNAL_WEIGHTS["rating_mismatch"],
            float(mismatch),
        )
        if mismatch:
            evidence.append("Rating conflicts with words in the review")

    total_weight = sum(weight for weight, _ in components.values())
    score = sum(weight * value for weight, value in components.values()) / total_weight
    if score >= 0.65:
        category = "High risk — prioritize review"
    elif score >= 0.35:
        category = "Review recommended"
    else:
        category = "Lower risk — not verified as genuine"

    return {
        "text_score": model_score,
        "risk_score": score,
        "category": category,
        "evidence": evidence or ["No configured risk signals were triggered"],
    }
