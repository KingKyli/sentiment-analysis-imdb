from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from keras.datasets import imdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "sentiment_pipeline.pkl"
TRAINING_SUMMARY_PATH = PROJECT_ROOT / "model" / "training_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the deployable IMDB sentiment pipeline."
    )
    parser.add_argument("--num-words", type=int, default=10000, help="Vocabulary size used when loading IMDB.")
    parser.add_argument("--max-features", type=int, default=20000, help="Maximum TF-IDF features.")
    parser.add_argument("--c", type=float, default=2.0, help="Inverse regularization strength for Logistic Regression.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum Logistic Regression iterations.")
    return parser.parse_args()


def decode_review(token_ids: list[int], reverse_word_index: dict[int, str]) -> str:
    tokens = []
    for token_id in token_ids:
        if token_id in (0, 1, 2):
            continue
        tokens.append(reverse_word_index.get(token_id - 3, "<UNK>"))
    return " ".join(tokens)


def load_imdb_text_data(num_words: int) -> tuple[list[str], list[int], list[str], list[int]]:
    word_index = imdb.get_word_index()
    reverse_word_index = {index: word for word, index in word_index.items()}

    (x_train_raw, y_train), (x_test_raw, y_test) = imdb.load_data(num_words=num_words)
    x_train_text = [decode_review(review, reverse_word_index) for review in x_train_raw]
    x_test_text = [decode_review(review, reverse_word_index) for review in x_test_raw]

    return x_train_text, list(y_train), x_test_text, list(y_test)


def build_pipeline(max_features: int, c_value: float, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            ("clf", LogisticRegression(max_iter=max_iter, C=c_value)),
        ]
    )


def main() -> None:
    args = parse_args()

    print("Loading IMDB dataset and decoding reviews...")
    x_train_text, y_train, x_test_text, y_test = load_imdb_text_data(args.num_words)

    print("Training TF-IDF + Logistic Regression pipeline...")
    pipeline = build_pipeline(args.max_features, args.c, args.max_iter)
    pipeline.fit(x_train_text, y_train)

    print("Evaluating trained pipeline on the held-out test split...")
    y_pred = pipeline.predict(x_test_text)
    summary = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "num_words": args.num_words,
        "max_features": args.max_features,
        "c": args.c,
        "max_iter": args.max_iter,
        "model": "TF-IDF + Logistic Regression",
        "dataset": "IMDB",
        "train_samples": len(y_train),
        "test_samples": len(y_test),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(pipeline, model_file)

    with TRAINING_SUMMARY_PATH.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved training summary to: {TRAINING_SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
