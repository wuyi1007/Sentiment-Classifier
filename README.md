# Linear Sentiment Classification

**Author:** Yibo Su

A binary sentiment classifier implemented from scratch using **logistic regression** and **stochastic gradient descent (SGD)**. The project explores how unigram, bigram, negation, and intensification features affect sentiment classification performance.

## Overview

The classifier predicts whether a sentence expresses **negative (`0`)** or **positive (`1`)** sentiment. Sentences are represented as sparse feature vectors using `Counter`, and feature names are mapped to integer indices through an `Indexer`. Logistic regression weights are stored in a NumPy array and optimized with SGD.

The project includes an always-positive baseline and three feature extraction modes:

- **`UNIGRAM`** — Lowercased word occurrence counts.
- **`BIGRAM`** — Binary features for adjacent word pairs.
- **`BETTER`** — Combined unigram/bigram features with negation and intensification features.

For negation, words following terms such as `not`, `no`, and `never` receive features like `Negated=good` until a punctuation or adversative boundary. For intensification, a word following terms such as `very` or `extremely` receives a feature like `Intensified=good`.

## Project Structure

```text
.
├── models.py                 # Feature extractors, logistic regression classifier, SGD training
├── sentiment_classifier.py   # Command-line entry point, evaluation, prediction output
├── sentiment_data.py         # Dataset loading and sentiment example utilities
├── utils.py                  # Shared utilities, including the feature Indexer
├── HW1_writeup.pdf           # Implementation details and experimental results
└── data/
    ├── train.txt             # 6,920 labeled training examples
    ├── dev.txt               # 872 labeled development examples
    └── test-blind.txt        # 1,821 unlabeled test examples
```

## Requirements

- Python 3
- NumPy

Install NumPy if necessary:

```bash
python -m pip install numpy
```

## Usage

Run these commands from the project directory.

**Unigram features:**

```bash
python sentiment_classifier.py --model LR --feats UNIGRAM
```

**Bigram features:**

```bash
python sentiment_classifier.py --model LR --feats BIGRAM
```

**Combined features (unigrams, bigrams, negation, and intensification):**

```bash
python sentiment_classifier.py --model LR --feats BETTER
```

**Trivial baseline (always predicts positive):**

```bash
python sentiment_classifier.py --model TRIVIAL
```

To skip generating blind-test predictions, add `--no_run_on_test`. By default, predictions are written to `test-blind.output.txt`. Training and development evaluation prints accuracy, precision, recall, and F1 for the positive class.

> **Implementation note:** Although the CLI defines `--lr` and `--num_epochs`, the `train_logistic_regression` function in the submitted `models.py` explicitly sets the learning rate to **0.015** and the number of epochs to **15**. Changing those CLI flags alone does not change these settings.

## Training Method

For a sparse feature vector \(x\), the model calculates a linear score and converts it to a probability:

\[
 p(y=1\mid x)=\sigma(w^\top x+b)
\]

SGD updates the weights and bias using the difference between the true label and predicted probability. The training examples are shuffled each epoch with a fixed random seed (`0`).

## Experimental Results

The following results are from `HW1_writeup.pdf`:

| Feature set | Train accuracy | Dev accuracy | Dev F1 |
|---|---:|---:|---:|
| Unigram | 91.68% | 78.33% | 77.42% |
| Bigram | 99.06% | 72.59% | 74.92% |
| Unigram + Bigram | 99.22% | 79.01% | 78.94% |
| Unigram + Bigram + Negation | 99.34% | 79.36% | 79.17% |
| Unigram + Bigram + Intensification | 99.22% | 79.01% | 78.94% |
| **Unigram + Bigram + Negation + Intensification** | **99.34%** | **79.59%** | **79.40%** |

The combined feature set achieved the best reported development accuracy (**79.59%**) and F1 (**79.40%**). Bigram-only features produced substantially higher training accuracy than development accuracy, suggesting overfitting. Adding negation improved development performance; intensification alone did not, but the combination of both features performed best.

> The table summarizes experiments documented in the write-up. The submitted `BETTER` feature extractor implements the final combined feature set, not a command-line switch for each intermediate ablation.

## Notes

- The model uses sparse, hand-engineered textual features rather than pretrained embeddings or neural networks.
- The blind test set is unlabeled, so this repository does not report blind-test accuracy.
- `sentiment_classifier.py` is the provided assignment driver and remains unchanged.
