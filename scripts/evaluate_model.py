from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from keras.datasets import imdb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "sentiment_pipeline.pkl"
METRICS_PATH = PROJECT_ROOT / "model" / "metrics.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "docs" / "assets" / "confusion_matrix.png"


def decode_review(token_ids: list[int], reverse_word_index: dict[int, str]) -> str:
    tokens = []
    for token_id in token_ids:
        if token_id in (0, 1, 2):
            continue
        tokens.append(reverse_word_index.get(token_id - 3, "<UNK>"))
    return " ".join(tokens)


def load_text_test_split() -> tuple[list[str], list[int]]:
    word_index = imdb.get_word_index()
    reverse_word_index = {index: word for word, index in word_index.items()}
    (_, _), (x_test_raw, y_test) = imdb.load_data(num_words=10000)
    x_test_text = [decode_review(review, reverse_word_index) for review in x_test_raw]
    return x_test_text, list(y_test)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing model artifact. Export the pipeline first to model/sentiment_pipeline.pkl."
        )

    with MODEL_PATH.open("rb") as model_file:
        pipeline = pickle.load(model_file)

    x_test_text, y_test = load_text_test_split()
    y_pred = pipeline.predict(x_test_text)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "support": int(len(y_test)),
        "model": "TF-IDF + Logistic Regression",
        "dataset": "IMDB",
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["negative", "positive"],
            output_dict=True,
            digits=4,
        ),
    }

    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Negative", "Positive"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Oranges", colorbar=False, values_format="d")
    ax.set_title("IMDB Sentiment Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
