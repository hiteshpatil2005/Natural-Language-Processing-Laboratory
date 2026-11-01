import os
import re
import time
import csv
import random
import warnings
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------- SETTINGS ---------------------------------
CSV_FILE = "news_articles.csv"
TEXT_COLUMN = "title"
LABEL_COLUMN = "category"
TEST_SIZE = 0.20
RANDOM_STATE = 42
EPOCHS = 5
BATCH_SIZE = 32
MAX_VOCAB_SIZE = 12000
MAX_SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
LEARNING_RATE = 0.001

# Change to True only if you want to fine-tune DistilBERT as well.
RUN_DISTILBERT = False
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_EPOCHS = 2
BERT_BATCH_SIZE = 8
BERT_MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(text):
    """Basic text cleaning while retaining words and numbers."""
    text = str(text).lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    return re.sub(r"\\s+", " ", text).strip()


def load_dataset():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find '{CSV_FILE}'. Put your existing CSV in this folder."
        )

    data = pd.read_csv(path)
    missing = [c for c in (TEXT_COLUMN, LABEL_COLUMN) if c not in data.columns]
    if missing:
        raise ValueError(
            f"Missing CSV column(s): {missing}. Required columns are "
            f"'{TEXT_COLUMN}' and '{LABEL_COLUMN}'."
        )

    data = data[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
    data[TEXT_COLUMN] = data[TEXT_COLUMN].astype(str).map(clean_text)
    data[LABEL_COLUMN] = data[LABEL_COLUMN].astype(str).str.strip()
    data = data[(data[TEXT_COLUMN] != "") & (data[LABEL_COLUMN] != "")]
    if len(data) < 10:
        print(
            f"\nWARNING: Only {len(data)} valid rows were found. "
            "This is too little data for a reliable train/test evaluation."
        )
        print(
            "Small-dataset mode will train on all available rows and report "
            "training accuracy only. These scores must not be treated as "
            "real-world model performance."
        )

    print(f"Loaded {len(data)} rows from {CSV_FILE}")
    print("\nCategory distribution:")
    print(data[LABEL_COLUMN].value_counts().to_string())
    return data


def make_vocabulary(texts):
    counts = Counter()
    for text in texts:
        counts.update(text.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counts.most_common(MAX_VOCAB_SIZE - 2):
        vocab[word] = len(vocab)
    return vocab


def encode_text(text, vocab):
    ids = [vocab.get(word, vocab["<UNK>"]) for word in text.split()]
    ids = ids[:MAX_SEQUENCE_LENGTH]
    ids += [vocab["<PAD>"]] * (MAX_SEQUENCE_LENGTH - len(ids))
    return ids


class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.x = torch.tensor([encode_text(t, vocab) for t in texts], dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, number_of_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM, HIDDEN_DIM, batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(HIDDEN_DIM * 2, number_of_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # Last forward and backward hidden states
        combined = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.output(self.dropout(combined))


def train_lstm(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"train_loss": [], "test_accuracy": []}
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        actual, predicted = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs.to(DEVICE))
                predicted.extend(outputs.argmax(dim=1).cpu().tolist())
                actual.extend(labels.tolist())

        average_loss = total_loss / max(1, len(train_loader))
        accuracy = accuracy_score(actual, predicted)
        history["train_loss"].append(average_loss)
        history["test_accuracy"].append(accuracy)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | Test accuracy: {accuracy:.4f}"
        )

    training_seconds = time.perf_counter() - start
    return history, training_seconds


def evaluate_lstm(model, test_loader, label_encoder):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(DEVICE))
            predicted.extend(outputs.argmax(dim=1).cpu().tolist())
            actual.extend(labels.tolist())

    print("\nBiLSTM classification report:")
    print(classification_report(
        actual, predicted, labels=list(range(len(label_encoder.classes_))),
        target_names=label_encoder.classes_, zero_division=0
    ))
    print("BiLSTM confusion matrix:")
    print(confusion_matrix(actual, predicted))

    # Measure average single-item model inference time (model already loaded).
    sample = next(iter(test_loader))[0][0:1].to(DEVICE)
    timings = []
    with torch.no_grad():
        for _ in range(30):
            start = time.perf_counter()
            model(sample)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    latency = sum(timings) / len(timings)
    return accuracy_score(actual, predicted), latency


def save_lstm_plots(history):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], marker="o")
    plt.title("BiLSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assignment9_lstm_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history["test_accuracy"]) + 1),
             history["test_accuracy"], marker="o")
    plt.title("BiLSTM Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assignment9_lstm_accuracy.png")
    plt.close()


def run_distilbert(train_texts, test_texts, train_labels, test_labels, label_encoder):
    """Optional real DistilBERT fine-tuning; requires transformers."""
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            Trainer, TrainingArguments
        )
    except ImportError:
        print("\nDistilBERT skipped: install it with: pip install transformers")
        return None

    class BertDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.encodings = tokenizer(
                list(texts), truncation=True, padding=True,
                max_length=BERT_MAX_LENGTH
            )
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {key: torch.tensor(value[index])
                    for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
            return item

    print("\nLoading DistilBERT tokenizer and pretrained model...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=len(label_encoder.classes_)
    )

    train_ds = BertDataset(train_texts, train_labels, tokenizer)
    test_ds = BertDataset(test_texts, test_labels, tokenizer)

    args = TrainingArguments(
        output_dir="assignment9_distilbert_output",
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none",
        seed=RANDOM_STATE
    )

    def metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=1)
        return {"accuracy": accuracy_score(labels, predictions)}

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds,
        eval_dataset=test_ds, compute_metrics=metrics
    )

    start = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - start
    result = trainer.predict(test_ds)
    predictions = result.predictions.argmax(axis=1)
    accuracy = accuracy_score(test_labels, predictions)
    print("\nDistilBERT classification report:")
    print(classification_report(
        test_labels, predictions,
        labels=list(range(len(label_encoder.classes_))),
        target_names=label_encoder.classes_, zero_division=0
    ))

    model.to(DEVICE)
    model.eval()
    one = tokenizer(
        test_texts.iloc[0], return_tensors="pt",
        truncation=True, max_length=BERT_MAX_LENGTH
    )
    one = {k: v.to(DEVICE) for k, v in one.items()}
    timings = []
    with torch.no_grad():
        for _ in range(10):
            start = time.perf_counter()
            model(**one)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    latency = sum(timings) / len(timings)
    return {
        "model": "Fine-tuned DistilBERT",
        "accuracy": accuracy,
        "training_seconds": training_seconds,
        "latency_ms": latency
    }


def main():
    warnings.filterwarnings("ignore")
    set_seed(RANDOM_STATE)
    print("=" * 68)
    print("ASSIGNMENT 9: TEXT CLASSIFICATION USING DEEP LEARNING")
    print("Bi-directional LSTM and optional DistilBERT")
    print("=" * 68)
    print(f"Running on: {DEVICE}")

    data = load_dataset()
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data[LABEL_COLUMN])

    # With fewer than 10 rows (or only one item in a class), a reliable
    # stratified test split is impossible. Use all rows for a demonstration
    # training run and clearly label the score as training accuracy.
    small_data_mode = (
        len(data) < 10 or min(Counter(encoded_labels).values()) < 2
    )

    if small_data_mode:
        train_texts = data[TEXT_COLUMN].reset_index(drop=True)
        train_labels = encoded_labels
        test_texts = train_texts
        test_labels = train_labels
        print("\nSMALL-DATASET MODE: using all rows for demonstration.")
        print("No independent test set is available; metrics are training metrics.")
    else:
        counts = Counter(encoded_labels)
        stratify = encoded_labels if min(counts.values()) >= 2 else None
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            data[TEXT_COLUMN], encoded_labels,
            test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
        )

    print("\nClasses:", ", ".join(label_encoder.classes_))
    print(f"Training rows: {len(train_texts)} | Testing/evaluation rows: {len(test_texts)}")

    vocab = make_vocabulary(train_texts)
    train_ds = NewsDataset(train_texts.tolist(), train_labels, vocab)
    test_ds = NewsDataset(test_texts.tolist(), test_labels, vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BiLSTMClassifier(len(vocab), len(label_encoder.classes_)).to(DEVICE)
    history, training_seconds = train_lstm(model, train_loader, test_loader)
    lstm_accuracy, lstm_latency = evaluate_lstm(model, test_loader, label_encoder)
    save_lstm_plots(history)

    metric_name = "training_accuracy" if small_data_mode else "accuracy"
    results = [{
        "model": "Bi-directional LSTM (PyTorch)",
        metric_name: lstm_accuracy,
        "accuracy": lstm_accuracy,
        "training_seconds": training_seconds,
        "latency_ms": lstm_latency
    }]

    if RUN_DISTILBERT and not small_data_mode:
        bert_result = run_distilbert(
            train_texts, test_texts, train_labels, test_labels, label_encoder
        )
        if bert_result:
            results.append(bert_result)
    elif RUN_DISTILBERT and small_data_mode:
        print("\nDistilBERT fine-tuning skipped: four records are insufficient "
              "for a meaningful train/test experiment.")

    print("\n" + "=" * 78)
    print("MODEL BENCHMARK RESULTS (measured on this computer)")
    print("Accuracy is training accuracy in small-dataset mode; otherwise test accuracy.")
    print(f"{'Model':<38} {'Accuracy':>10} {'Train (s)':>12} {'Latency (ms)':>14} {'<200 ms':>9}")
    print("-" * 78)
    for result in results:
        passed = "YES" if result["latency_ms"] < 200 else "NO"
        print(
            f"{result['model']:<38} "
            f"{result['accuracy'] * 100:>9.2f}% "
            f"{result['training_seconds']:>12.2f} "
            f"{result['latency_ms']:>14.2f} {passed:>9}"
        )

    with open("assignment9_model_benchmark.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=[
                "model", "accuracy", "training_accuracy",
                "training_seconds", "latency_ms"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nFiles saved:")
    print("- assignment9_lstm_loss.png")
    print("- assignment9_lstm_accuracy.png")
    print("- assignment9_model_benchmark.csv")
    print("\nAssignment 9 completed.")


if __name__ == "__main__":
    main()

































"""
ASSIGNMENT 9: TEXT CLASSIFICATION USING DEEP LEARNING (LSTM / BERT)
Multi-class news classification: World, Sports, Business, Sci/Tech

DATA FILE
---------
Keep your existing news_articles.csv in the same folder as this Python file.
Required CSV columns:
    title       : news article title/text
    category    : category label (World, Sports, Business, Sci/Tech)

No sample CSV data is created by this program.

WHAT THIS PROGRAM DOES
----------------------
1. Loads and checks your external CSV file.
2. Splits data into training and testing sets.
3. Trains a Bi-directional LSTM text classifier using PyTorch.
4. Optionally fine-tunes DistilBERT when the Transformers packages are installed
   and RUN_DISTILBERT is set to True.
5. Reports accuracy, training time, average inference latency and whether
   the 200 ms target is met.
6. Saves LSTM loss/accuracy curves and a benchmark CSV.

INSTALLATION
------------
Windows (PowerShell / CMD):
    py -3.13 -m venv nlp_env
    nlp_env\\Scripts\\activate
    python -m pip install --upgrade pip
    pip install pandas scikit-learn matplotlib torch

Ubuntu / Linux:
    python3 -m venv nlp_env
    source nlp_env/bin/activate
    python -m pip install --upgrade pip
    pip install pandas scikit-learn matplotlib torch

Optional packages for DistilBERT:
    pip install transformers

Note: PyTorch installation can vary by CPU/GPU and operating system. If the
standard command fails, use the installation selector at https://pytorch.org/get-started/locally/

RUN
---
    python Assignment9_Text_Classification_LSTM_BERT.py

IMPORTANT
---------
- First run of DistilBERT downloads the pretrained model and needs internet.
- DistilBERT fine-tuning can be slow on CPU. Set RUN_DISTILBERT = False to run
  the BiLSTM part only.
- The 200 ms requirement is measured on your computer; it is not guaranteed.
- Accuracy is calculated from your own labelled CSV test split, not assumed.
"""


# --------------------------- VIVA QUESTIONS ----------------------------
"""
1. What is text classification?
   It is the task of assigning a predefined category to a text.

2. What is deep learning?
   A machine-learning approach that uses neural networks with multiple layers.

3. What is an RNN?
   A recurrent neural network processes sequential data while maintaining state.

4. What is an LSTM?
   Long Short-Term Memory is an RNN architecture designed to learn long-range
   dependencies using gates and a cell state.

5. Why is LSTM used for text?
   It can process words in sequence and retain useful contextual information.

6. What is a Bi-directional LSTM?
   It reads a sequence in both forward and backward directions.

7. What is an embedding?
   A dense numeric vector representing a word or token.

8. What is a vocabulary?
   The set of unique tokens mapped to integer IDs.

9. What is padding?
   Adding pad tokens so sequences in a batch have equal length.

10. What is an unknown token?
    A special token used for a word not present in the training vocabulary.

11. What is an epoch?
    One complete pass through the training dataset.

12. What is a batch?
    A group of training examples processed together.

13. What is a loss function?
    A function measuring the difference between predictions and true labels.

14. What is CrossEntropyLoss?
    A loss function commonly used for multi-class classification.

15. What is an optimizer?
    An algorithm that updates model parameters to reduce loss.

16. What is backpropagation?
    Computing gradients of the loss with respect to model parameters.

17. What is dropout?
    A regularization technique that randomly disables some units during training.

18. What is overfitting?
    When a model learns training data too closely and performs poorly on new data.

19. What is a Transformer?
    A neural architecture based primarily on attention mechanisms.

20. What is BERT?
    Bidirectional Encoder Representations from Transformers, a pretrained language model.

21. What is DistilBERT?
    A smaller, faster Transformer model distilled from BERT.

22. What is fine-tuning?
    Further training a pretrained model on a task-specific dataset.

23. What is tokenization?
    Splitting text into tokens and converting them into model inputs.

24. What is a train-test split?
    Dividing data into training data and unseen testing data.

25. What is accuracy?
    The fraction of predictions that are correct.

26. What is a confusion matrix?
    A table comparing actual classes with predicted classes.

27. What is inference latency?
    The time taken by a trained model to make a prediction.

28. Why is inference speed important?
    Fast predictions are useful in real-time applications.

29. What does the 200 ms target mean?
    The average measured prediction time should be below 200 milliseconds.

30. Why compare LSTM and DistilBERT?
    To compare a recurrent neural network with a pretrained Transformer on
    accuracy, training cost and prediction speed.
"""
