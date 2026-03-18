import json
import pickle
import re
from pathlib import Path

from deep_translator import GoogleTranslator
from langdetect import DetectorFactory, LangDetectException, detect
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IMDB Sentiment Demo", layout="wide")

DetectorFactory.seed = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "sentiment_pipeline.pkl"
METRICS_PATH = PROJECT_ROOT / "model" / "metrics.json"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "model" / "model_comparison.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "docs" / "assets" / "confusion_matrix.png"
AUTHOR_LABEL = "Portfolio-ready NLP project"
AUTHOR_CONTEXT = "Built around model comparison, deployable sentiment analysis, multilingual inference handling, and reproducible evaluation artifacts."
DEFAULT_METRICS = {
    "accuracy": 0.8838,
    "precision": 0.8843,
    "recall": 0.8830,
    "f1_score": 0.8837,
    "model": "TF-IDF + Logistic Regression",
    "dataset": "IMDB",
    "support": 25000,
}
DEFAULT_COMPARISON = {
    "selected_model": "TF-IDF + Logistic Regression",
    "selection_reason": "Logistic Regression remains the deployable default because it stays close to the strongest benchmark while keeping probability outputs for the app.",
    "transformer_reference_reason": "DistilBERT is tracked as a stronger reference benchmark, while the app keeps the lighter classical pipeline unless the project is intentionally upgraded to transformer deployment.",
    "models": [
        {
            "model": "TF-IDF + Multinomial Naive Bayes",
            "benchmark_type": "trained_in_repo",
            "accuracy": 0.8559,
            "precision": 0.8640,
            "recall": 0.8449,
            "f1_score": 0.8543,
            "training_seconds": 13.161,
            "inference_ms_per_sample": 0.2396,
            "evaluation_samples": 25000,
        },
        {
            "model": "TF-IDF + Logistic Regression",
            "benchmark_type": "trained_in_repo",
            "accuracy": 0.8838,
            "precision": 0.8843,
            "recall": 0.8830,
            "f1_score": 0.8837,
            "training_seconds": 14.666,
            "inference_ms_per_sample": 0.2418,
            "evaluation_samples": 25000,
        },
        {
            "model": "TF-IDF + Linear SVM",
            "benchmark_type": "trained_in_repo",
            "accuracy": 0.8618,
            "precision": 0.8694,
            "recall": 0.8514,
            "f1_score": 0.8603,
            "training_seconds": 18.708,
            "inference_ms_per_sample": 0.2357,
            "evaluation_samples": 25000,
        },
        {
            "model": "DistilBERT transformer benchmark",
            "benchmark_type": "pretrained_transformer",
            "accuracy": 0.9600,
            "precision": 0.9298,
            "recall": 1.0000,
            "f1_score": 0.9636,
            "training_seconds": None,
            "inference_ms_per_sample": 188.6365,
            "evaluation_samples": 100,
        },
    ],
}
LANGUAGE_LABELS = {
    "en": "English",
    "el": "Greek",
    "unknown": "Unknown",
}
EXAMPLES = {
    "EN Pos": "This movie was surprisingly moving, beautifully acted, and worth every minute.",
    "EN Neg": "The plot was a mess, the acting felt flat, and I regret finishing it.",
    "GR Pos": "Πολύ καλή ταινία με εξαιρετικές ερμηνείες και έντονο συναίσθημα.",
    "GR Neg": "Η ταινία ήταν κουραστική, προβλέψιμη και με απογοητευτικό τέλος.",
    "Mixed": "The performances were strong, but the story dragged and the ending felt rushed.",
}


@st.cache_resource
def load_pipeline():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)
    return DEFAULT_METRICS


@st.cache_data
def load_model_comparison():
    if MODEL_COMPARISON_PATH.exists():
        with MODEL_COMPARISON_PATH.open("r", encoding="utf-8") as comparison_file:
            return json.load(comparison_file)
    return DEFAULT_COMPARISON


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def contains_greek(text):
    return bool(re.search(r"[\u0370-\u03FF\u1F00-\u1FFF]", text))


def detect_input_language(text):
    if contains_greek(text):
        return "el"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def translate_to_english(text):
    return GoogleTranslator(source="auto", target="en").translate(text)


def prepare_inference_text(raw_text):
    normalized_text = normalize_text(raw_text)
    language_code = detect_input_language(normalized_text)
    translation_used = language_code not in {"en", "unknown"}
    translated_text = None

    if translation_used:
        translated_text = normalize_text(translate_to_english(normalized_text))
        model_text = translated_text
    else:
        model_text = normalized_text

    return {
        "original_text": normalized_text,
        "model_text": model_text,
        "translated_text": translated_text,
        "language_code": language_code,
        "language_label": LANGUAGE_LABELS.get(language_code, language_code.upper()),
        "translation_used": translation_used,
    }


def format_metric(value):
    return f"{float(value) * 100:.2f}%"


def format_duration(seconds):
    if seconds is None:
        return "N/A"
    return f"{float(seconds):.2f}s"


def format_latency(milliseconds):
    return f"{float(milliseconds):.4f} ms"


def build_chart_frame(comparison_data):
    rows = []
    for item in comparison_data.get("models", []):
        rows.append(
            {
                "Model": item.get("model", "Unknown"),
                "Accuracy": round(float(item.get("accuracy", 0.0)), 4),
                "F1": round(float(item.get("f1_score", 0.0)), 4),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Accuracy", "F1"])
    chart_frame = pd.DataFrame(rows).set_index("Model")
    return chart_frame[["Accuracy", "F1"]]


def render_metric_card(label, value):
    st.markdown(
        f"""
        <div class=\"metric-card\">
            <div class=\"metric-label\">{label}</div>
            <div class=\"metric-value\">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #f6efe7 0%, #fffaf5 45%, #f3f6fb 100%);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-card, .panel-card, .result-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(34, 40, 49, 0.08);
        border-radius: 24px;
        box-shadow: 0 18px 50px rgba(25, 35, 52, 0.08);
        padding: 1.4rem;
        backdrop-filter: blur(14px);
    }
    .hero-card {
        padding: 1.8rem;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 3rem;
        line-height: 1.05;
        font-weight: 700;
        color: #1f2430;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #4f566b;
        max-width: 760px;
        margin-bottom: 1rem;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 0.75rem;
    }
    .chip {
        background: #1f2430;
        color: #fffaf5;
        border-radius: 999px;
        padding: 0.42rem 0.8rem;
        font-size: 0.9rem;
        letter-spacing: 0.01em;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 650;
        color: #222831;
        margin-bottom: 0.25rem;
    }
    .section-copy {
        color: #586173;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(180deg, #fff 0%, #f8f3ee 100%);
        border: 1px solid rgba(34, 40, 49, 0.08);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.75rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6a7283;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #202531;
    }
    .result-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.5rem 0.85rem;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }
    .badge-positive {
        background: #d8f5df;
        color: #14532d;
    }
    .badge-negative {
        background: #ffe0da;
        color: #8a1c0f;
    }
    .result-copy {
        color: #4f566b;
        margin-bottom: 0.8rem;
    }
    .supporting-copy {
        color: #667085;
        font-size: 0.95rem;
        line-height: 1.65;
    }
    .footer-card {
        background: rgba(31, 36, 48, 0.96);
        border-radius: 24px;
        padding: 1.3rem 1.4rem;
        color: #f5f1eb;
        box-shadow: 0 18px 50px rgba(25, 35, 52, 0.16);
    }
    .footer-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .footer-copy {
        color: rgba(245, 241, 235, 0.84);
        line-height: 1.65;
        font-size: 0.95rem;
        margin-bottom: 0.9rem;
    }
    .footer-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
    }
    .footer-chip {
        border: 1px solid rgba(255, 250, 245, 0.14);
        border-radius: 999px;
        padding: 0.38rem 0.7rem;
        color: #fffaf5;
        font-size: 0.88rem;
    }
    div[data-testid="column"] .stButton > button {
        min-height: 2.75rem;
        white-space: nowrap;
        font-size: 0.92rem;
        padding-left: 0.35rem;
        padding-right: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">IMDB Sentiment Analysis</div>
        <div class="hero-subtitle">
            A portfolio-ready NLP demo that turns raw movie reviews into live sentiment predictions using a deployed TF-IDF + Logistic Regression pipeline with multilingual input handling.
        </div>
        <div class="chip-row">
            <div class="chip">Interactive Demo</div>
            <div class="chip">IMDB Dataset</div>
            <div class="chip">TF-IDF + Logistic Regression</div>
            <div class="chip">DistilBERT Benchmark</div>
            <div class="chip">Greek + English Input</div>
            <div class="chip">88.38% Test Accuracy</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Run the export cell from the notebook first so that "
        "model/sentiment_pipeline.pkl is created."
    )
    st.stop()

pipeline = load_pipeline()
metrics = load_metrics()
comparison = load_model_comparison()

left_col, right_col = st.columns([1.45, 0.9], gap="large")

with left_col:
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-title">Try the live demo</div>
            <div class="section-copy">Paste a review in English or Greek. Greek input is translated to English before inference so the deployed model stays aligned with its training data.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    example_columns = st.columns(len(EXAMPLES))
    for column, label in zip(example_columns, EXAMPLES):
        if column.button(label, width="stretch"):
            st.session_state.review_text = EXAMPLES[label]

    st.text_area(
        "Write a movie review",
        key="review_text",
        height=220,
        placeholder="Type a movie review in English or Greek...",
    )

    predict_clicked = st.button("Predict sentiment", type="primary")
    cleaned_text = st.session_state.review_text.strip()

    if predict_clicked:
        st.session_state.prediction_result = None
        if not cleaned_text:
            st.warning("Enter a review before running prediction.")
        else:
            try:
                prepared = prepare_inference_text(cleaned_text)
                prediction = pipeline.predict([prepared["model_text"]])[0]
                label = "Positive" if prediction == 1 else "Negative"
                probabilities = None
                confidence = None

                if hasattr(pipeline, "predict_proba"):
                    probabilities = pipeline.predict_proba([prepared["model_text"]])[0]
                    confidence = float(max(probabilities))

                st.session_state.prediction_result = {
                    "label": label,
                    "prediction": int(prediction),
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "input_length": len(prepared["original_text"].split()),
                    "model_length": len(prepared["model_text"].split()),
                    "original_text": prepared["original_text"],
                    "model_text": prepared["model_text"],
                    "translated_text": prepared["translated_text"],
                    "translation_used": prepared["translation_used"],
                    "language_label": prepared["language_label"],
                }
            except Exception as exc:
                st.error(
                    "Translation-aware inference failed. If you entered Greek text, verify internet access and retry. "
                    f"Details: {exc}"
                )

    st.markdown(
        """
        <div class="panel-card" style="margin-top: 1rem;">
            <div class="section-title">Why this feels stronger than a notebook</div>
            <div class="section-copy">
                The app accepts raw text, handles multilingual input responsibly, runs a serialized pipeline, and returns a production-style prediction flow that is easier to demonstrate in interviews and on a CV.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    st.markdown(
        """
        <div class="result-card">
            <div class="section-title">Model snapshot</div>
            <div class="section-copy">Final deployable model used in the demo.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    model_info_col, dataset_info_col = st.columns(2)
    with model_info_col:
        render_metric_card("Pipeline", metrics.get("model", DEFAULT_METRICS["model"]))
    with dataset_info_col:
        render_metric_card("Dataset", metrics.get("dataset", DEFAULT_METRICS["dataset"]))

    metrics_cols = st.columns(2)
    metric_items = [
        ("Accuracy", format_metric(metrics.get("accuracy", DEFAULT_METRICS["accuracy"]))),
        ("Precision", format_metric(metrics.get("precision", DEFAULT_METRICS["precision"]))),
        ("Recall", format_metric(metrics.get("recall", DEFAULT_METRICS["recall"]))),
        ("F1-score", format_metric(metrics.get("f1_score", DEFAULT_METRICS["f1_score"]))),
    ]
    for column, (label, value) in zip(metrics_cols * 2, metric_items):
        with column:
            render_metric_card(label, value)

    st.caption(f"Evaluated on {int(metrics.get('support', DEFAULT_METRICS['support'])):,} held-out IMDB reviews.")

    st.markdown(
        """
        <div class="panel-card" style="margin-top: 0.6rem;">
            <div class="section-title">Live prediction</div>
            <div class="section-copy">Run the model on your review and inspect language handling, confidence, and class probabilities.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result = st.session_state.prediction_result
    if result is None:
        st.info("Run a prediction to populate the analysis panel.")
    else:
        badge_class = "badge-positive" if result["prediction"] == 1 else "badge-negative"
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-badge {badge_class}">{result['label']} review</div>
                <div class="result-copy">Detected input language: {result['language_label']}. Model processed {result['model_length']} tokens.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result["translation_used"]:
            st.info("Non-English input was translated to English before inference so the classifier stays aligned with its training data.")
            st.text_area("Translated model input", value=result["model_text"], height=120, disabled=True)
        else:
            st.caption("Input was scored directly without translation.")

        if result["confidence"] is not None:
            st.metric("Confidence", f"{result['confidence']:.1%}")
            if result["confidence"] < 0.65:
                st.warning("Low-confidence prediction. Treat this result as directional rather than definitive.")

        if result["probabilities"] is not None:
            negative_score = float(result["probabilities"][0])
            positive_score = float(result["probabilities"][1])
            st.write("Negative probability")
            st.progress(negative_score)
            st.caption(f"{negative_score:.1%}")
            st.write("Positive probability")
            st.progress(positive_score)
            st.caption(f"{positive_score:.1%}")

    st.markdown(
        """
        <div class="panel-card" style="margin-top: 0.8rem;">
            <div class="section-title">Limitations</div>
            <div class="section-copy">
                The classifier is trained on English IMDB reviews. English input is native, while Greek input is translated before inference. Mixed sentiment, sarcasm, short text, and translation noise can still reduce reliability.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Evaluation Evidence")
evaluation_col, notes_col = st.columns([1.05, 0.95], gap="large")

with evaluation_col:
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-title">Confusion matrix</div>
            <div class="section-copy">Held-out test set behavior for the final deployable pipeline.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if CONFUSION_MATRIX_PATH.exists():
        st.image(str(CONFUSION_MATRIX_PATH), width="stretch")
    else:
        st.info("Confusion matrix artifact not found yet. Re-run the evaluation export cell in the notebook if needed.")

with notes_col:
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-title">Operational notes</div>
            <div class="supporting-copy">
                <strong>English reviews:</strong> scored directly by the deployed pipeline.<br><br>
                <strong>Greek reviews:</strong> translated to English before inference for better alignment with the IMDB training distribution.<br><br>
                <strong>Professional takeaway:</strong> this demo now shows model performance, deployable artifacts, and a realistic multilingual inference layer instead of relying on notebook-only outputs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Model Comparison")
comparison_col, rationale_col = st.columns([1.15, 0.85], gap="large")

with comparison_col:
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-title">Benchmark table</div>
            <div class="section-copy">Held-out IMDB comparison across classical baselines and stronger reference models.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    comparison_rows = []
    for item in comparison.get("models", []):
        comparison_rows.append(
            {
                "Model": item.get("model", "Unknown"),
                "Type": item.get("benchmark_type", "trained_in_repo"),
                "Accuracy": format_metric(item.get("accuracy", 0.0)),
                "Precision": format_metric(item.get("precision", 0.0)),
                "Recall": format_metric(item.get("recall", 0.0)),
                "F1": format_metric(item.get("f1_score", 0.0)),
                "Train": format_duration(item.get("training_seconds")),
                "Infer / sample": format_latency(item.get("inference_ms_per_sample", 0.0)),
                "Eval samples": f"{int(item.get('evaluation_samples', 0)):,}",
            }
        )
    st.dataframe(comparison_rows, width="stretch", hide_index=True)
    st.bar_chart(
        build_chart_frame(comparison),
        color=["#cb6e17", "#205fa8"],
        height=320,
        width="stretch",
    )
    st.caption("Accuracy and F1 are plotted together so model quality gaps are visible without reading the full table.")

with rationale_col:
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-title">Selection rationale</div>
            <div class="section-copy">Why the demo keeps its current deployable model after benchmarking stronger alternatives.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_metric_card("Selected model", comparison.get("selected_model", DEFAULT_COMPARISON["selected_model"]))
    st.write(comparison.get("selection_reason", DEFAULT_COMPARISON["selection_reason"]))
    if comparison.get("transformer_reference_reason"):
        st.write(comparison["transformer_reference_reason"])
    st.caption("The benchmark is designed to show both model quality and deployment trade-offs, not just the single highest score.")

st.markdown(
    f"""
    <div class="footer-card" style="margin-top: 1.4rem;">
        <div class="footer-title">About this build</div>
        <div class="footer-copy">
            {AUTHOR_LABEL}. {AUTHOR_CONTEXT}
        </div>
        <div class="footer-chip-row">
            <div class="footer-chip">Streamlit demo</div>
            <div class="footer-chip">IMDB dataset</div>
            <div class="footer-chip">Benchmark: classical + transformer</div>
            <div class="footer-chip">Greek-to-English translation layer</div>
            <div class="footer-chip">Metrics + confusion matrix</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
