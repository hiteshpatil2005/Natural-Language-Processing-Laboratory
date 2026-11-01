from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)


def load_dataset():
    """Import the user's existing CSV file. No dataset is created here."""
    csv_path = Path(__file__).resolve().parent / "fmcg_product_reviews.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            "CSV file not found. Place fmcg_product_reviews.csv "
            "in the same folder as this Python file."
        )

    data = pd.read_csv(csv_path)

    required_columns = {"review", "sentiment"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            "CSV must contain these columns: review and sentiment."
        )

    data = data[["review", "sentiment"]].dropna()
    data["review"] = data["review"].astype(str).str.strip()
    data = data[data["review"] != ""]

    data["sentiment"] = pd.to_numeric(data["sentiment"], errors="coerce")
    data = data.dropna(subset=["sentiment"])
    data["sentiment"] = data["sentiment"].astype(int)

    if not set(data["sentiment"].unique()).issubset({0, 1}):
        raise ValueError("Sentiment labels must be 0 (Negative) or 1 (Positive).")

    if data["sentiment"].nunique() != 2:
        raise ValueError("The dataset must contain both sentiment classes: 0 and 1.")

    if len(data) < 4:
        raise ValueError("At least 4 valid reviews are required.")

    return data


def train_and_evaluate(texts, labels):
    """Train three ML classifiers and evaluate them on test data."""
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels
    )

    # Learn vocabulary and TF-IDF weights from training data only.
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        roc_auc = roc_auc_score(y_test, probabilities)

        results[name] = {
            "accuracy": accuracy,
            "confusion_matrix": matrix,
            "roc_auc": roc_auc,
            "predictions": predictions,
            "model": model
        }

    return results, y_test


def monitor_brand_health(predictions, threshold_percent=30.0):
    """Alert if the percentage of negative predictions exceeds threshold."""
    total = len(predictions)

    if total == 0:
        return "NO DATA"

    negative_count = sum(1 for value in predictions if value == 0)
    negative_percent = (negative_count / total) * 100

    if negative_percent > threshold_percent:
        return (
            f"CRITICAL ALERT: Negative sentiment is {negative_percent:.1f}% "
            f"(above {threshold_percent:.1f}%)"
        )

    return f"NORMAL: Negative sentiment is {negative_percent:.1f}%"


def main():
    print("=" * 65)
    print("ASSIGNMENT 7: SENTIMENT ANALYSIS USING MACHINE LEARNING")
    print("=" * 65)

    try:
        data = load_dataset()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print("\nDataset error:", error)
        return

    reviews = data["review"].tolist()
    labels = data["sentiment"].tolist()

    print(f"\nSuccessfully imported {len(reviews)} reviews from CSV.")
    print("Sentiment labels: 0 = Negative, 1 = Positive")

    results, y_test = train_and_evaluate(reviews, labels)

    print("\n--- Model Evaluation Results ---")
    print(f"{'Model':<28} {'Accuracy':<12} {'ROC-AUC':<10}")
    print("-" * 53)

    for name, result in results.items():
        print(
            f"{name:<28} "
            f"{result['accuracy'] * 100:>7.2f}%     "
            f"{result['roc_auc']:.4f}"
        )

    print("\n--- Confusion Matrices ---")
    print("Rows = Actual labels [Negative, Positive]")
    print("Columns = Predicted labels [Negative, Positive]")

    for name, result in results.items():
        print(f"\n{name}:")
        print(result["confusion_matrix"])

    print("\n--- Logistic Regression Classification Report ---")
    lr_predictions = results["Logistic Regression"]["predictions"]
    print(
        classification_report(
            y_test,
            lr_predictions,
            labels=[0, 1],
            target_names=["Negative", "Positive"],
            zero_division=0
        )
    )

    print("--- FMCG Brand Health Monitoring ---")
    alert = monitor_brand_health(lr_predictions, threshold_percent=30.0)
    print("Status:", alert)

    print("\nAssignment 7 completed.")


if __name__ == "__main__":
    main()








































"""
ASSIGNMENT 7: SENTIMENT ANALYSIS USING MACHINE LEARNING

Main tasks:
1. Read product reviews and sentiment labels from an external CSV file.
2. Convert review text into TF-IDF features.
3. Train Logistic Regression, Multinomial Naive Bayes and SVM.
4. Evaluate models using Accuracy, Confusion Matrix and ROC-AUC.
5. Monitor negative sentiment and generate a brand health alert.

IMPORTANT:
- The CSV dataset is NOT included in this Python file.
- Keep your existing fmcg_product_reviews.csv in the same folder.
- Required CSV column names: review, sentiment
- sentiment must contain binary labels: 0 = Negative, 1 = Positive.

------------------------------------------------------------
INSTALLATION COMMANDS
------------------------------------------------------------

WINDOWS (Command Prompt / VS Code Terminal):
    python --version
    python -m pip install --upgrade pip
    python -m pip install pandas scikit-learn
    python Assignment7_Sentiment_Analysis.py

UBUNTU / LINUX:
    python3 --version
    sudo apt update
    sudo apt install python3 python3-pip
    python3 -m pip install --user pandas scikit-learn
    python3 Assignment7_Sentiment_Analysis.py

CSV FILE:
    Keep this structure in one folder:

    Assignment7_Sentiment_Analysis.py
    fmcg_product_reviews.csv

    Example CSV header:
    review,sentiment

    The review and sentiment columns must be present.
"""

# ============================================================
# VIVA QUESTIONS AND ANSWERS
# ============================================================

"""
1. What is sentiment analysis?
   Sentiment analysis identifies the opinion or emotion expressed in text,
   commonly positive, negative or neutral.

2. What is the objective of this assignment?
   To classify FMCG product reviews as positive or negative using ML models
   and monitor the percentage of negative reviews.

3. What is machine learning?
   Machine learning enables computers to learn patterns from data and make
   predictions without explicitly programming every rule.

4. What is supervised learning?
   It is learning from labelled examples where the correct output is known.

5. What is a feature?
   A feature is an input variable used by a machine learning model.

6. What is a label?
   A label is the target value that the model is trained to predict.

7. What is TF-IDF?
   TF-IDF means Term Frequency-Inverse Document Frequency. It gives higher
   importance to useful words and reduces the importance of very common words.

8. Why is TF-IDF used for reviews?
   It converts text reviews into numerical features that ML models can process.

9. What is a classifier?
   A classifier is a model that assigns input data to predefined categories.

10. What is Logistic Regression?
    It is a classification algorithm that estimates the probability of a class.

11. What is Naive Bayes?
    It is a probabilistic classifier based on Bayes' theorem and a feature
    independence assumption.

12. What is Multinomial Naive Bayes?
    It is a Naive Bayes variant commonly used for discrete text features,
    including word counts and non-negative TF-IDF values.

13. What is SVM?
    Support Vector Machine is an algorithm that finds a decision boundary
    that separates classes with a maximum margin.

14. What is a training dataset?
    It is the part of the dataset used to train the model.

15. What is a testing dataset?
    It is separate data used to evaluate how well the trained model performs.

16. Why do we split data into training and testing sets?
    To estimate model performance on unseen data.

17. What is train_test_split()?
    It is a scikit-learn function that divides data into training and testing sets.

18. What does random_state do?
    It makes the random split reproducible.

19. What is stratify in train_test_split()?
    It helps preserve the class proportions in both training and testing sets.

20. What is accuracy?
    Accuracy is the fraction of all predictions that are correct.

21. What is a confusion matrix?
    It is a table comparing actual class labels with predicted class labels.

22. What are TP, TN, FP and FN?
    TP: positive correctly predicted; TN: negative correctly predicted;
    FP: negative incorrectly predicted as positive;
    FN: positive incorrectly predicted as negative.

23. What is ROC-AUC?
    ROC-AUC measures how well a classifier distinguishes between two classes
    across different classification thresholds.

24. What is precision?
    Precision is the proportion of predicted positive cases that are actually
    positive.

25. What is recall?
    Recall is the proportion of actual positive cases correctly identified.

26. What is the F1-score?
    It is the harmonic mean of precision and recall.

27. What is brand health monitoring?
    It monitors customer sentiment to identify a possible increase in negative
    reviews and alert the business.

28. What is the alert threshold in this assignment?
    The threshold is 30 percent. An alert is generated when the negative
    prediction percentage is greater than 30 percent.

29. Why is the CSV file used?
    It stores product review text and sentiment labels in tabular form so
    the program can import and process the dataset.

30. Why should TF-IDF be fitted only on training data?
    To avoid data leakage. The test data must not influence the vocabulary
    or IDF weights learned during training.

NOTE:
This is an educational sentiment-analysis program. The brand alert is based
on model predictions from the test split, not a live production monitoring
system. Model scores depend on the user's CSV dataset.
"""
