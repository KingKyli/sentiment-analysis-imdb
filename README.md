# IMDB Sentiment Analysis

An end-to-end sentiment analysis project built on the IMDB movie review dataset.

This repository upgrades a university ML assignment into a portfolio-ready project by adding:
- a reproducible project structure
- an interactive Streamlit demo
- a deployable sentiment pipeline for raw text input
- explicit model benchmarking across baseline and stronger linear classifiers
- multilingual inference support for English and Greek input
- clear setup and execution instructions

## Result Snapshot

- Deployable TF-IDF + Logistic Regression pipeline accuracy: `0.8838` on the IMDB test set
- Precision: `0.8843`, Recall: `0.8830`, F1-score: `0.8837`
- Benchmark comparison artifact is stored in `model/model_comparison.json`
- Transformer reference benchmark artifact is stored in `model/transformer_benchmark.json`
- Evaluation artifacts are stored in `model/metrics.json` and `docs/assets/confusion_matrix.png`
- Error analysis artifact is stored in `docs/project_artifacts/error_analysis.json`

## App Showcase

### Overview

![App Overview](docs/project_artifacts/screenshots/app_overview.png)

### Greek Prediction Flow

This example shows Greek input being detected, translated to English, and then scored by the deployed classifier.

![Greek Prediction](docs/project_artifacts/screenshots/app_greek_prediction.png)

### English Prediction Flow

This example shows the model's native English inference path without translation.

![English Prediction](docs/project_artifacts/screenshots/app_english_prediction.png)

### Evaluation Evidence

The app also embeds evaluation artifacts directly into the interface so the demo is backed by measurable model evidence.

![Evaluation Evidence](docs/project_artifacts/screenshots/app_evaluation.png)

## Results

Final deployable model:
- Model: `TF-IDF + Logistic Regression`
- Dataset: `IMDB`
- Test samples: `25,000`

Performance summary:
- Accuracy: `88.38%`
- Precision: `88.43%`
- Recall: `88.30%`
- F1-score: `88.37%`

### Model Comparison

The project now includes two layers of benchmark evidence:

1. Full classical benchmark on the held-out IMDB split
2. Transformer reference benchmark on a smaller CPU-friendly subset for practical portfolio comparison

| Model | Metrics source | Practical role |
| --- | --- | --- |
| TF-IDF + Multinomial Naive Bayes | `model/model_comparison.json` | lightweight lexical baseline |
| TF-IDF + Logistic Regression | `model/model_comparison.json` | strong linear baseline with probability outputs |
| TF-IDF + Linear SVM | `model/model_comparison.json` | stronger margin-based benchmark |
| DistilBERT transformer benchmark | `model/transformer_benchmark.json` | stronger modern reference model |

Final-model selection is now explicit rather than implicit:
- the classical training script compares Naive Bayes, Logistic Regression, and Linear SVM on accuracy, precision, recall, and F1
- it also records training time and per-sample inference cost
- it keeps Logistic Regression as the deployed model only when it stays close to the best benchmark while preserving confidence scores for the app
- if another benchmark wins by a clearly larger F1 margin, the exported deployable pipeline switches automatically

This makes the project easier to defend technically because the final model is chosen after reproducible comparison and documented trade-offs.

Latest benchmark outcome from `model/model_comparison.json`:
- TF-IDF + Multinomial Naive Bayes: Accuracy `85.59%`, F1 `85.43%`
- TF-IDF + Logistic Regression: Accuracy `88.38%`, F1 `88.37%`
- TF-IDF + Linear SVM: Accuracy `86.18%`, F1 `86.03%`
- DistilBERT transformer benchmark: Accuracy `96.00%`, F1 `96.36%` on `100` held-out reviews, with much higher inference cost
- Final deployable choice: Logistic Regression, because it remains fast, lightweight, probability-based, and far easier to deploy inside the Streamlit demo

Important comparison note:
- the transformer benchmark is intentionally marked as a reference benchmark, not the deployed model
- it was evaluated on a smaller `100`-review subset because this repository is CPU-first and the goal is to show stronger-model awareness without turning the project into a heavyweight serving stack
- this keeps the project honest: the stronger model is acknowledged, but the final production choice is still justified by deployment trade-offs

Confusion matrix:

![Confusion Matrix](docs/assets/confusion_matrix.png)

## Error Analysis

Advanced touch added: reproducible error analysis on the highest-confidence mistakes made by the final deployed model.

Artifact:
- `docs/project_artifacts/error_analysis.json`

Observed failure patterns:
- strongly positive keywords inside otherwise negative reviews can trigger false positives
- mixed or contrastive reviews often confuse the classifier, especially when sentiment shifts late in the text
- reviews that say a film is "not my kind of movie" or start negatively and end positively can produce false negatives
- short emphatic statements can be misread when the lexical signal is stronger than the actual label

## Why This Project Stands Out

- moves beyond notebook-only experimentation into a usable product demo
- lets a reviewer interact with the model through raw text input
- shows both ML understanding and practical packaging for deployment
- is easy to explain in a CV, interview, or portfolio walkthrough

## Problem

The goal is to classify IMDB movie reviews as positive or negative and present the final result through an interface that is easy for a non-technical reviewer to test.

## Approach

The repository combines two layers of work:

1. Academic experimentation in the notebook
	- custom Naive Bayes and Logistic Regression implementations
	- evaluation with precision, recall, and F1-score
	- learning curve analysis and model comparisons

2. Deployable inference pipeline for the demo
	- IMDB reviews decoded back into raw text
	- TF-IDF vectorization with unigram and bigram features
	- benchmark comparison across Multinomial Naive Bayes, Logistic Regression, and Linear SVM
	- DistilBERT added as a stronger transformer reference benchmark
	- final deployable model selected with an explicit quality-versus-usability rule
	- serialized pipeline loaded directly by the Streamlit app

## Project Structure

```text
sentiment-analysis-imdb/
├── .streamlit/
│   └── config.toml
├── runtime.txt
├── scripts/
│   ├── train_pipeline.py
│   └── evaluate_model.py
├── app/
│   └── app.py
├── model/
│   └── sentiment_pipeline.pkl
├── notebooks/
│   └── assignment.ipynb
├── docs/
│   ├── assets/
│   └── project_artifacts/
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Demo

Live demo:

- `https://sentiment-analysis-imdb-kingkyli.streamlit.app`

Run locally:

```bash
streamlit run app/app.py
```

Then enter a movie review and get an instant sentiment prediction.

Example:
- Input: `This movie was amazing, emotional, and visually stunning.`
- Output: `Positive`

Greek example:
- Input: `Πολύ καλή ταινία με εξαιρετικές ερμηνείες.`
- Processing: auto-translated to English before inference
- Output: `Positive`

The app also displays:
- prediction confidence
- class probabilities
- model summary metrics
- benchmark comparison across candidate models
- chart-based comparison of Accuracy and F1
- detected input language
- translated model input for non-English reviews
- embedded confusion matrix and evaluation evidence
- project framing that makes the demo easier to present

For reproducible evaluation outside the interface, see `model/metrics.json` and `docs/assets/confusion_matrix.png`.
For reproducible model selection evidence, see `model/model_comparison.json`.
For the transformer reference benchmark, see `model/transformer_benchmark.json`.

## Multilingual Input Handling

The deployed classifier is trained on English IMDB reviews, so English remains the model's native inference language.

To improve usability for Greek input, the app now:
- detects the input language automatically
- translates Greek reviews to English before inference
- shows the translated text used by the model
- warns when prediction confidence is low

This is a more honest and production-like solution than pretending the underlying classifier is natively multilingual.

## How To Export The Model

The Streamlit app expects a serialized pipeline at `model/sentiment_pipeline.pkl`.

Recommended reproducible workflow:

```bash
python scripts/train_pipeline.py
python scripts/evaluate_model.py
```

This produces:
- `model/sentiment_pipeline.pkl`
- `model/training_summary.json`
- `model/model_comparison.json`
- `model/metrics.json`
- `docs/assets/confusion_matrix.png`

Optional stronger-reference benchmark:

```bash
python scripts/benchmark_transformer.py --max-samples 100 --batch-size 8
```

This additionally produces:
- `model/transformer_benchmark.json`
- an updated `model/model_comparison.json` with the transformer entry merged in

1. Open the notebook inside `notebooks/assignment.ipynb`
2. Run the new export section at the end of the notebook
3. Confirm that `model/sentiment_pipeline.pkl` has been created

The notebook now also contains an evaluation cell that exports:
- `model/metrics.json`
- `docs/assets/confusion_matrix.png`

## What The Project Shows

- classical NLP baselines on IMDB
- custom implementations of Naive Bayes and Logistic Regression
- reproducible comparison with library-based Naive Bayes, Logistic Regression, and Linear SVM
- a stronger DistilBERT transformer reference benchmark with explicit deployment trade-off analysis
- learning curves and evaluation metrics
- a usable application layer for demonstration purposes
- conversion of notebook work into a deployable inference pipeline

## Setup

Runtime app dependencies:

```bash
pip install -r requirements.txt
```

Full local environment for training and evaluation scripts:

```bash
pip install -r requirements-dev.txt
```

## How To Run

```bash
streamlit run app/app.py
```

Open the local Streamlit URL in your browser, paste a review, and inspect the returned sentiment and confidence scores.

Note: Greek input support uses runtime translation, so an internet connection is required when scoring non-English text.

`localhost` is normal for local development and demonstrations. If you want a more public-facing presentation later, the natural next step is to deploy the app to Streamlit Community Cloud or another hosting platform.

## Deployment

The repository is now structured to be deployment-ready for Streamlit Community Cloud.

Recommended deployment target:
- repository root: `sentiment-analysis-imdb`
- app entrypoint: `app/app.py`
- dependencies: `requirements.txt`
- optional local training extras: `requirements-dev.txt`
- Streamlit config: `.streamlit/config.toml`
- Python runtime: `runtime.txt`

Typical deployment flow:

1. Push the repository to GitHub
2. Sign in to Streamlit Community Cloud
3. Create a new app from the repository
4. Set the main file path to `app/app.py`
5. Deploy and wait for the build to finish

Public app URL:
- `https://sentiment-analysis-imdb-kingkyli.streamlit.app`

## Interview-Ready Talking Points

- started from a university assignment and rebuilt it into something that can be used and evaluated outside the notebook
- benchmarked Naive Bayes, Logistic Regression, and Linear SVM before deciding which pipeline to deploy
- added a DistilBERT transformer benchmark as a stronger modern reference model
- kept or switched the final deployable model based on measured F1 trade-offs and app usability constraints
- kept the project grounded in proper evaluation while also making the demo usable for someone non-technical
- handled Greek input through translation before inference so the live app stays consistent with an English-trained model
- packaged the model in a simple Streamlit interface so anyone can test it without setting up the development environment

## CV Bullets

- Built an IMDB sentiment analysis project end to end, from model training and evaluation to a deployed Streamlit demo.
- Benchmarked Naive Bayes, Logistic Regression, and Linear SVM, then documented the final model choice with reproducible comparison artifacts.
- Added a DistilBERT transformer benchmark to compare a stronger modern NLP model against the lightweight deployed pipeline.
- Added Greek-to-English preprocessing for live inputs and documented performance with metrics, confusion matrix, and error analysis.
- Turned a notebook-based university assignment into a clean GitHub project with reproducible scripts, deployment-ready structure, and a public demo.

## Interview Pitch

This project started as a university assignment on sentiment analysis for the IMDB dataset. I wanted to take it beyond the notebook, so I benchmarked Naive Bayes, Logistic Regression, and Linear SVM on a shared held-out split, added a DistilBERT transformer reference benchmark, exported the chosen deployable pipeline, wrapped it in a Streamlit app, added Greek input support through translation, and included evaluation artifacts such as metrics, a confusion matrix, and error analysis. The goal was not just to get a good score, but to present the work in a way that is reproducible, usable, and easy to discuss in a professional setting.
