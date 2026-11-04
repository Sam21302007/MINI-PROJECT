import html
import hashlib
import os
import re

import joblib
import pandas as pd
import streamlit as st

from image_ocr import ImageOCRError, extract_review_text
from review_monitor import explain_text, score_review


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_COUNT = 10
MAX_CSV_ROWS = 5000

st.set_page_config(page_title="Review Integrity Monitor", page_icon="🔎", layout="wide")


@st.cache_resource
def load_model_and_vectorizer():
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(VECTORIZER_PATH):
        from train import main as train_model

        train_model()
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def parse_optional_bool(value):
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "verified"}:
        return True
    if normalized in {"false", "0", "no", "unverified"}:
        return False
    raise ValueError(f"Unrecognized verified_purchase value: {value!r}")


def optional_metadata(row):
    metadata = {}
    if "verified_purchase" in row:
        metadata["verified_purchase"] = parse_optional_bool(row["verified_purchase"])
    for column in ("account_age_days", "reviews_last_hour", "rating"):
        if column in row and not pd.isna(row[column]):
            metadata[column] = float(row[column])
    return metadata


def score_review_rows(rows, model, vectorizer):
    history = []
    scored_rows = []
    for row in rows:
        text = str(row["text"]).strip()
        explanation = explain_text(text, model, vectorizer)
        result = score_review(
            text,
            model,
            vectorizer,
            history=history,
            metadata=optional_metadata(row),
        )
        scored_rows.append({
            **row,
            "text_model_score": result["text_score"],
            "triage_score": result["risk_score"],
            "triage_category": result["category"],
            "risk_signals": "; ".join(result["evidence"]),
            "fake_words": ", ".join(explanation["fake_words"]),
            "genuine_words": ", ".join(explanation["genuine_words"]),
        })
        history.append({"text": text})
    return pd.DataFrame(scored_rows)


def render_highlighted_review(text, fake_words, genuine_words):
    word_styles = {word.casefold(): "fake" for word in fake_words}
    word_styles.update({
        word.casefold(): "genuine"
        for word in genuine_words
        if word.casefold() not in word_styles
    })
    if not word_styles:
        st.text(text)
        return

    pattern = re.compile(
        r"(?<!\w)(" + "|".join(
            re.escape(word)
            for word in sorted(word_styles, key=len, reverse=True)
        ) + r")(?!\w)",
        flags=re.IGNORECASE,
    )
    chunks = []
    cursor = 0
    for match in pattern.finditer(text):
        chunks.append(html.escape(text[cursor:match.start()]))
        kind = word_styles[match.group().casefold()]
        color = "#7f1d1d" if kind == "fake" else "#14532d"
        background = "#fee2e2" if kind == "fake" else "#dcfce7"
        chunks.append(
            f'<mark style="background:{background};color:{color};'
            f'padding:2px 4px;border-radius:4px">{html.escape(match.group())}</mark>'
        )
        cursor = match.end()
    chunks.append(html.escape(text[cursor:]))
    st.html(
        '<div style="white-space:pre-wrap;line-height:1.8">'
        + "".join(chunks)
        + "</div>"
    )


def render_review_result(record, index=1):
    score = float(record["text_model_score"])
    genuine_score = 1.0 - score
    title = record.get("source_file", f"Review {index}")
    st.subheader(f"Result: {title}")
    metric_cols = st.columns(2)
    metric_cols[0].metric("Suspicious / fake score", f"{score:.0%}")
    metric_cols[1].metric("Genuine / real score", f"{genuine_score:.0%}")
    st.progress(score)
    if score >= 0.5:
        st.warning("The model leans suspicious. This is a review signal, not proof of fraud.")
    else:
        st.success("The model leans genuine. This is not verification that the review is real.")

    st.markdown("**Review text with influential words highlighted**")
    fake_words = [word.strip() for word in str(record.get("fake_words", "")).split(",") if word.strip()]
    genuine_words = [
        word.strip() for word in str(record.get("genuine_words", "")).split(",") if word.strip()
    ]
    render_highlighted_review(str(record["text"]), fake_words, genuine_words)
    legend_fake, legend_genuine = st.columns(2)
    legend_fake.markdown("🔴 Red highlight: words that pushed the model toward suspicious")
    legend_genuine.markdown("🟢 Green highlight: words that pushed it toward genuine")
    st.caption(
        "Highlights show influential words from this trained model, not causal evidence. "
        "The bundled model was trained on 70 synthetic demonstration examples."
    )
    if record.get("risk_signals"):
        st.caption(f"Other triage signals: {record['risk_signals']}")


def render_results_panel(results, key_prefix):
    if results.empty:
        return
    records = results.to_dict("records")
    fake_count = sum(float(row["text_model_score"]) >= 0.5 for row in records)
    genuine_count = len(records) - fake_count

    heading_col, download_col = st.columns([4, 1], vertical_alignment="bottom")
    with heading_col:
        st.header("Review result")
        st.caption(f"{len(records)} review(s) analyzed · {fake_count} suspicious · {genuine_count} genuine-leaning")
    with download_col:
        st.download_button(
            "Download CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name=f"{key_prefix}_review_triage.csv",
            mime="text/csv",
            key=f"download_{key_prefix}_results",
        )

    render_review_result(records[0])
    if len(records) > 1:
        with st.expander(f"Other results ({len(records) - 1})"):
            for index, record in enumerate(records[1:], start=2):
                render_review_result(record, index=index)
    with st.expander("Full results table"):
        st.dataframe(results, width="stretch", hide_index=True)


st.title("Review integrity monitor")
st.markdown(
    "Analyze review screenshots with local OCR, or upload review data as a CSV. "
    "Results include the main suspicious/genuine score and model-influential word highlights."
)
st.warning(
    "The bundled model was trained on 70 synthetic demonstration reviews, not verified "
    "real-world labels. Treat scores as experimental triage suggestions—not proof that "
    "a review is fake or genuine."
)

model, vectorizer = load_model_and_vectorizer()
if model is None or vectorizer is None:
    st.error("No trained model found. From the project folder, run `python train.py` first.")
    st.stop()

image_tab, csv_tab = st.tabs(["Review screenshots", "Review data (CSV)"])

with image_tab:
    st.subheader("Analyze review screenshots")
    st.caption(
        "Upload up to 10 PNG/JPG/WebP images, ideally one review per image. OCR runs "
        "locally using Tesseract. Check and correct the extracted text before scoring."
    )
    uploaded_images = st.file_uploader(
        "Choose review screenshots",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="review_images",
        max_upload_size=10,
    )

    image_records = []
    invalid_images = []
    for uploaded in uploaded_images or []:
        image_bytes = uploaded.getvalue()
        if len(image_bytes) > MAX_IMAGE_BYTES:
            invalid_images.append(f"{uploaded.name}: file exceeds 10 MB.")
            continue
        image_records.append({
            "name": uploaded.name,
            "bytes": image_bytes,
            "id": hashlib.sha256(image_bytes).hexdigest(),
        })
    if len(image_records) > MAX_IMAGE_COUNT:
        st.error(f"Upload at most {MAX_IMAGE_COUNT} images at a time.")
        image_records = []
    for message in invalid_images:
        st.error(message)

    if image_records and st.button("Extract text from images", type="primary"):
        extracted = []
        with st.spinner("Reading text from uploaded images on this computer..."):
            for record in image_records:
                try:
                    text = extract_review_text(record["bytes"])
                except ImageOCRError as error:
                    st.error(f"{record['name']}: {error}")
                else:
                    extracted.append({**record, "text": text})
        st.session_state["ocr_reviews"] = extracted

    ocr_reviews = st.session_state.get("ocr_reviews", [])
    if ocr_reviews:
        st.info(
            "OCR can misread text. Each image is treated as one review; a screenshot "
            "containing multiple reviews may need to be split or edited manually."
        )
        corrected_reviews = []
        for index, record in enumerate(ocr_reviews, start=1):
            st.image(record["bytes"], caption=record["name"], alt=f"Uploaded review screenshot {index}", width=360)
            corrected_text = st.text_area(
                f"Extracted text — {record['name']}",
                value=record["text"],
                height=120,
                key=f"ocr_text_{record['id']}",
            )
            corrected_reviews.append({
                "source_file": record["name"],
                "text": corrected_text.strip(),
            })

        if st.button("Classify extracted reviews", key="classify_images"):
            valid_reviews = [row for row in corrected_reviews if row["text"]]
            if not valid_reviews:
                st.warning("No review text was extracted. Correct or enter text above first.")
            else:
                results = score_review_rows(valid_reviews, model, vectorizer)
                signature = hashlib.sha256(
                    "\n".join(row["text"] for row in valid_reviews).encode("utf-8")
                ).hexdigest()
                st.session_state["image_results"] = {
                    "signature": signature,
                    "data": results.to_dict("records"),
                }

    if ocr_reviews:
        corrected_reviews = [
            {
                "source_file": record["name"],
                "text": st.session_state.get(
                    f"ocr_text_{record['id']}",
                    record["text"],
                ).strip(),
            }
            for record in ocr_reviews
        ]
        valid_reviews = [row for row in corrected_reviews if row["text"]]
        current_signature = hashlib.sha256(
            "\n".join(row["text"] for row in valid_reviews).encode("utf-8")
        ).hexdigest()
        saved = st.session_state.get("image_results")
        if saved and saved["signature"] == current_signature:
            render_results_panel(pd.DataFrame(saved["data"]), "image")

with csv_tab:
    st.subheader("Analyze review data")
    st.caption(
        "Upload a CSV with a `text` column. Optional columns: `rating`, "
        "`verified_purchase`, `account_age_days`, and `reviews_last_hour`."
    )
    uploaded_csv = st.file_uploader(
        "Choose a review CSV",
        type=["csv"],
        key="review_csv",
        max_upload_size=20,
    )
    if uploaded_csv is not None:
        try:
            batch = pd.read_csv(uploaded_csv)
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f"Could not read the CSV: {error}")
        else:
            if "text" not in batch.columns:
                st.error("CSV must contain a `text` column.")
            elif batch.empty:
                st.warning("The uploaded CSV has no rows.")
            elif len(batch) > MAX_CSV_ROWS:
                st.error(f"Upload at most {MAX_CSV_ROWS} reviews at a time.")
            elif batch["text"].isna().any() or batch["text"].astype(str).str.strip().eq("").any():
                st.error("The `text` column contains blank values. Remove them and upload again.")
            else:
                st.caption(f"{len(batch)} reviews ready for scoring.")
                upload_signature = hashlib.sha256(uploaded_csv.getvalue()).hexdigest()
                if st.button("Classify CSV reviews", type="primary"):
                    try:
                        results = score_review_rows(batch.to_dict("records"), model, vectorizer)
                    except (TypeError, ValueError) as error:
                        st.error(f"Invalid review metadata: {error}")
                    else:
                        st.session_state["csv_results"] = {
                            "signature": upload_signature,
                            "data": results.to_dict("records"),
                        }
                saved = st.session_state.get("csv_results")
                if saved and saved["signature"] == upload_signature:
                    render_results_panel(pd.DataFrame(saved["data"]), "csv")

st.caption(
    "All image OCR and review scoring run locally. Uploaded content is not sent to an "
    "external review API. Scores are not calibrated and should only help prioritize "
    "human review."
)
