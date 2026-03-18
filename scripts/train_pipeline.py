from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

from keras.datasets import imdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "sentiment_pipeline.pkl"
TRAINING_SUMMARY_PATH = PROJECT_ROOT / "model" / "training_summary.json"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "model" / "model_comparison.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the deployable IMDB sentiment pipeline."
    )
    parser.add_argument("--num-words", type=int, default=10000, help="Vocabulary size used when loading IMDB.")
    parser.add_argument("--max-features", type=int, default=20000, help="Maximum TF-IDF features.")
    parser.add_argument("--c", type=float, default=2.0, help="Inverse regularization strength for Logistic Regression.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum Logistic Regression iterations.")
    parser.add_argument(
        "--selection-tolerance",
        type=float,
        default=0.005,
        help="Keep Logistic Regression as the deployable model when its F1 is within this margin of the best model.",
    )
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


def build_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
    )


def build_logistic_pipeline(max_features: int, c_value: float, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", build_vectorizer(max_features)),
            ("clf", LogisticRegression(max_iter=max_iter, C=c_value, random_state=42)),
        ]
    )


def build_naive_bayes_pipeline(max_features: int) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", build_vectorizer(max_features)),
            ("clf", MultinomialNB()),
        ]
    )


def build_linear_svm_pipeline(max_features: int, c_value: float, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", build_vectorizer(max_features)),
            ("clf", LinearSVC(C=c_value, max_iter=max_iter, random_state=42)),
        ]
    )


def evaluate_pipeline(
    name: str,
    pipeline: Pipeline,
    x_train_text: list[str],
    y_train: list[int],
    x_test_text: list[str],
    y_test: list[int],
) -> tuple[dict[str, float | int | str | bool], Pipeline]:
    train_start = perf_counter()
    pipeline.fit(x_train_text, y_train)
    training_seconds = perf_counter() - train_start

    inference_start = perf_counter()
    y_pred = pipeline.predict(x_test_text)
    inference_seconds = perf_counter() - inference_start

    summary = {
        "model": name,
        "benchmark_type": "trained_in_repo",
        "model_family": "classical_linear" if "Naive Bayes" not in name else "classical_probabilistic",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "training_seconds": round(float(training_seconds), 3),
        "inference_ms_per_sample": round(float(inference_seconds / len(x_test_text) * 1000), 4),
        "evaluation_samples": len(x_test_text),
        "supports_predict_proba": bool(hasattr(pipeline, "predict_proba")),
        "deployment_ready": True,
        "notes": "Trained from scratch in this repository on the IMDB training split.",
    }
    return summary, pipeline


def choose_deployable_model(
    model_summaries: list[dict[str, float | int | str | bool]],
    selection_tolerance: float,
) -> tuple[dict[str, float | int | str | bool], str]:
    best_model = max(model_summaries, key=lambda item: float(item["f1_score"]))
    logistic_model = next(item for item in model_summaries if item["model"] == "TF-IDF + Logistic Regression")
    f1_gap = float(best_model["f1_score"]) - float(logistic_model["f1_score"])

    if f1_gap <= selection_tolerance:
        return (
            logistic_model,
            "Logistic Regression was kept as the final deployable model because its F1-score stayed close to the best benchmark while preserving probability outputs for confidence reporting in the app.",
        )

    return (
        best_model,
        "The strongest benchmark outperformed Logistic Regression by more than the allowed tolerance, so the final deployable model was switched to the highest-F1 pipeline.",
    )


def main() -> None:
    args = parse_args()

    print("Loading IMDB dataset and decoding reviews...")
    x_train_text, y_train, x_test_text, y_test = load_imdb_text_data(args.num_words)

    print("Training benchmark models on the IMDB text split...")
    model_builders = [
        ("TF-IDF + Multinomial Naive Bayes", build_naive_bayes_pipeline(args.max_features)),
        ("TF-IDF + Logistic Regression", build_logistic_pipeline(args.max_features, args.c, args.max_iter)),
        ("TF-IDF + Linear SVM", build_linear_svm_pipeline(args.max_features, args.c, args.max_iter)),
    ]

    trained_pipelines: dict[str, Pipeline] = {}
    model_summaries: list[dict[str, float | int | str | bool]] = []

    for model_name, pipeline in model_builders:
        print(f"Benchmarking {model_name}...")
        model_summary, trained_pipeline = evaluate_pipeline(
            model_name,
            pipeline,
            x_train_text,
            y_train,
            x_test_text,
            y_test,
        )
        model_summaries.append(model_summary)
        trained_pipelines[model_name] = trained_pipeline

    selected_model_summary, selection_reason = choose_deployable_model(
        model_summaries,
        args.selection_tolerance,
    )
    selected_model_name = str(selected_model_summary["model"])
    pipeline = trained_pipelines[selected_model_name]

    summary = {
        **selected_model_summary,
        "num_words": args.num_words,
        "max_features": args.max_features,
        "c": args.c,
        "max_iter": args.max_iter,
        "selection_tolerance": args.selection_tolerance,
        "dataset": "IMDB",
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "selection_reason": selection_reason,
        "benchmark_models": [item["model"] for item in model_summaries],
    }

    comparison = {
        "dataset": "IMDB",
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "selected_model": selected_model_name,
        "selection_reason": selection_reason,
        "selection_tolerance": args.selection_tolerance,
        "benchmark_scope": "Classical in-repo training benchmark with optional external transformer add-ons.",
        "models": model_summaries,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(pipeline, model_file)

    with TRAINING_SUMMARY_PATH.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    with MODEL_COMPARISON_PATH.open("w", encoding="utf-8") as comparison_file:
        json.dump(comparison, comparison_file, indent=2)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved training summary to: {TRAINING_SUMMARY_PATH}")
    print(f"Saved comparison artifact to: {MODEL_COMPARISON_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
