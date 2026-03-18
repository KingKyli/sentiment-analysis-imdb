from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from keras.datasets import imdb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_COMPARISON_PATH = PROJECT_ROOT / "model" / "model_comparison.json"
TRANSFORMER_BENCHMARK_PATH = PROJECT_ROOT / "model" / "transformer_benchmark.json"
DEFAULT_MODEL_NAME = "textattack/distilbert-base-uncased-imdb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a pretrained DistilBERT sentiment model on the IMDB test split."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model id used for the transformer benchmark.",
    )
    parser.add_argument(
        "--num-words",
        type=int,
        default=10000,
        help="Vocabulary size used when loading IMDB from keras.datasets.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Number of held-out IMDB reviews to evaluate. Use 0 to score the full 25,000-review test split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for transformer inference.",
    )
    return parser.parse_args()


def decode_review(token_ids: list[int], reverse_word_index: dict[int, str]) -> str:
    tokens = []
    for token_id in token_ids:
        if token_id in (0, 1, 2):
            continue
        tokens.append(reverse_word_index.get(token_id - 3, "<UNK>"))
    return " ".join(tokens)


def load_imdb_test_text(num_words: int, max_samples: int) -> tuple[list[str], list[int]]:
    word_index = imdb.get_word_index()
    reverse_word_index = {index: word for word, index in word_index.items()}
    (_, _), (x_test_raw, y_test) = imdb.load_data(num_words=num_words)

    x_test_text = [decode_review(review, reverse_word_index) for review in x_test_raw]
    labels = list(y_test)

    if max_samples > 0:
        return x_test_text[:max_samples], labels[:max_samples]
    return x_test_text, labels


def load_transformer_pipeline(model_name: str):
    from transformers import pipeline

    return pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        framework="pt",
        model_kwargs={"use_safetensors": False},
    )


def label_to_binary(label: str) -> int:
    normalized = label.strip().upper()
    if normalized in {"POSITIVE", "LABEL_1"}:
        return 1
    if normalized in {"NEGATIVE", "LABEL_0"}:
        return 0
    raise ValueError(f"Unsupported transformer label: {label}")


def benchmark_transformer(model_name: str, texts: list[str], y_true: list[int], batch_size: int) -> dict[str, float | int | str | bool | None]:
    sentiment_pipeline = load_transformer_pipeline(model_name)

    inference_start = perf_counter()
    outputs = sentiment_pipeline(texts, batch_size=batch_size, truncation=True, max_length=512)
    inference_seconds = perf_counter() - inference_start

    y_pred = [label_to_binary(item["label"]) for item in outputs]

    return {
        "model": "DistilBERT transformer benchmark",
        "benchmark_type": "pretrained_transformer",
        "model_family": "transformer",
        "source_model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred)), 4),
        "training_seconds": None,
        "inference_ms_per_sample": round(float(inference_seconds / len(texts) * 1000), 4),
        "evaluation_samples": len(texts),
        "supports_predict_proba": True,
        "deployment_ready": False,
        "notes": "Inference-only benchmark using a pretrained Hugging Face DistilBERT sentiment model. This is included as a stronger reference, not as the deployed app model.",
    }


def merge_into_model_comparison(transformer_summary: dict[str, float | int | str | bool | None]) -> dict[str, object]:
    if MODEL_COMPARISON_PATH.exists():
        with MODEL_COMPARISON_PATH.open("r", encoding="utf-8") as comparison_file:
            comparison = json.load(comparison_file)
    else:
        comparison = {
            "dataset": "IMDB",
            "selected_model": "TF-IDF + Logistic Regression",
            "selection_reason": "Transformer benchmark not yet compared against a trained in-repo baseline.",
            "models": [],
        }

    existing_models = [
        model for model in comparison.get("models", []) if model.get("model") != transformer_summary["model"]
    ]
    existing_models.append(transformer_summary)
    comparison["models"] = existing_models
    comparison["transformer_reference_model"] = transformer_summary["model"]
    comparison["transformer_reference_reason"] = (
        "DistilBERT is tracked as a stronger reference benchmark, while the app keeps the lighter classical pipeline unless the project is intentionally upgraded to transformer deployment."
    )
    return comparison


def main() -> None:
    args = parse_args()

    print("Loading IMDB held-out reviews for transformer benchmarking...")
    x_test_text, y_test = load_imdb_test_text(args.num_words, args.max_samples)

    print(f"Running transformer benchmark with {args.model_name} on {len(x_test_text)} reviews...")
    transformer_summary = benchmark_transformer(
        args.model_name,
        x_test_text,
        y_test,
        args.batch_size,
    )

    comparison = merge_into_model_comparison(transformer_summary)

    TRANSFORMER_BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRANSFORMER_BENCHMARK_PATH.open("w", encoding="utf-8") as benchmark_file:
        json.dump(transformer_summary, benchmark_file, indent=2)

    with MODEL_COMPARISON_PATH.open("w", encoding="utf-8") as comparison_file:
        json.dump(comparison, comparison_file, indent=2)

    print(f"Saved transformer benchmark to: {TRANSFORMER_BENCHMARK_PATH}")
    print(f"Updated comparison artifact: {MODEL_COMPARISON_PATH}")
    print(json.dumps(transformer_summary, indent=2))


if __name__ == "__main__":
    main()