import csv
from pathlib import Path


MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def load_tweets():
    """Import tweets and language codes from the user's CSV file."""
    csv_path = Path(__file__).resolve().parent / (
        "multilingual_opinion_tweets.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            "multilingual_opinion_tweets.csv was not found. "
            "Place your CSV file in the same folder as this Python file."
        )

    tweets = []
    languages = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"text", "lang"}
        if not reader.fieldnames or not required_columns.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError("CSV must contain 'text' and 'lang' columns.")

        for row in reader:
            text = (row.get("text") or "").strip()
            language = (row.get("lang") or "").strip().upper()

            if text:
                tweets.append(text)
                languages.append(language if language else "UNKNOWN")

    if not tweets:
        raise ValueError("The CSV file does not contain any usable tweet text.")

    return tweets, languages


def create_sentiment_classifier():
    """Load the pretrained multilingual sentiment model."""
    try:
        from transformers import pipeline
    except ImportError as error:
        raise ImportError(
            "Transformers is not installed. Run: "
            "python -m pip install transformers torch"
        ) from error

    print("\nLoading pretrained model:", MODEL_NAME)
    print("The first run may take time because the model must be downloaded.")

    # Limit CPU thread usage to avoid excessive thread overhead on some PCs.
    import torch
    torch.set_num_threads(2)

    return pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        device=-1
    )


def convert_label(label):
    """Convert the model's output labels into readable sentiment names."""
    label = label.upper()

    label_mapping = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }

    return label_mapping.get(label, label)


def predict_sentiment(classifier, texts):
    """Predict sentiment and confidence score for each text."""
    results = []

    # Process one tweet at a time. This is usually more manageable on CPU.
    for index, text in enumerate(texts, start=1):
        print(f"Analysing tweet {index}/{len(texts)}...")

        prediction = classifier(
            text,
            truncation=True,
            max_length=128
        )[0]

        results.append({
            "text": text,
            "label": convert_label(prediction["label"]),
            "score": round(float(prediction["score"]), 4)
        })

    return results


def main():
    print("=" * 75)
    print("ASSIGNMENT 8: TRANSFORMER-BASED MULTILINGUAL SENTIMENT ANALYSIS")
    print("=" * 75)

    try:
        tweets, languages = load_tweets()
        print(f"\nSuccessfully imported {len(tweets)} tweets from CSV.")

        classifier = create_sentiment_classifier()
        results = predict_sentiment(classifier, tweets)

        print("\n--- MULTILINGUAL SENTIMENT ANALYSIS RESULTS ---")
        print(f"{'Language':<12} {'Sentiment':<12} {'Confidence':<12} Tweet")
        print("-" * 100)

        for language, result in zip(languages, results):
            print(
                f"{language:<12} "
                f"{result['label']:<12} "
                f"{result['score']:<12.4f} "
                f"{result['text']}"
            )

        print("\n--- SENTIMENT SUMMARY ---")
        counts = {
            "Negative": 0,
            "Neutral": 0,
            "Positive": 0
        }

        for result in results:
            if result["label"] in counts:
                counts[result["label"]] += 1

        for sentiment, count in counts.items():
            print(f"{sentiment:<10}: {count}")

        print("\nMultilingual sentiment analysis completed.")

    except FileNotFoundError as error:
        print("\nCSV file error:", error)
    except ValueError as error:
        print("\nCSV data error:", error)
    except Exception as error:
        print("\nError during model loading or prediction:", error)
        print("Check your internet connection and installed libraries.")
        print("Also ensure that the model can be downloaded from Hugging Face.")


if __name__ == "__main__":
    main()




































"""
ASSIGNMENT 8: TRANSFORMER-BASED MULTILINGUAL SENTIMENT ANALYSIS
USING HUGGING FACE

Topics covered:
1. Hugging Face Transformers pipeline.
2. Pretrained XLM-RoBERTa multilingual sentiment model.
3. Sentiment classification: Negative, Neutral and Positive.
4. Confidence score for each prediction.
5. Import multilingual tweets/opinions from an external CSV file.

IMPORTANT:
- The CSV dataset is NOT included in this Python file.
- Keep your existing multilingual_opinion_tweets.csv in the same folder.
- Required CSV columns: text, lang
- text = tweet/opinion text
- lang = language code, for example EN, HI, FR.
- The pretrained model is downloaded by Hugging Face the first time
  the script runs. Internet access is required for the first download.

------------------------------------------------------------
INSTALLATION COMMANDS
------------------------------------------------------------

WINDOWS (Command Prompt / VS Code Terminal):
    python --version
    python -m pip install --upgrade pip
    python -m pip install transformers torch
    python Assignment8_Transformer_Sentiment.py

UBUNTU / LINUX:
    python3 --version
    sudo apt update
    sudo apt install python3 python3-pip
    python3 -m pip install --user transformers torch
    python3 Assignment8_Transformer_Sentiment.py

CSV FILE:
    Keep both files in the same folder:

    Assignment8_Transformer_Sentiment.py
    multilingual_opinion_tweets.csv

    Required CSV header:
    text,lang
"""


# ============================================================
# VIVA QUESTIONS AND ANSWERS
# ============================================================

"""
1. What is sentiment analysis?
   Sentiment analysis identifies the opinion or emotion expressed in text,
   such as positive, negative or neutral.

2. What is a Transformer?
   A Transformer is a neural network architecture that uses attention
   mechanisms to process sequences and understand context.

3. What is Hugging Face?
   Hugging Face is a platform and ecosystem that provides pretrained models,
   datasets and tools for machine learning and NLP.

4. What is the Transformers library?
   It is a Python library used to load and use pretrained Transformer models.

5. What is a pretrained model?
   A pretrained model is a model that has already learned patterns from
   a large dataset before being used for a particular task.

6. Which model is used in this assignment?
   cardiffnlp/twitter-xlm-roberta-base-sentiment.

7. What is XLM-RoBERTa?
   XLM-RoBERTa is a multilingual Transformer model trained to understand
   text in multiple languages.

8. What does multilingual mean?
   It means supporting or working with more than one language.

9. What is a Hugging Face pipeline?
   A pipeline is a high-level API that simplifies tasks such as sentiment
   analysis, text classification and named entity recognition.

10. What does pipeline("sentiment-analysis") do?
    It creates a sentiment-analysis pipeline that accepts text and returns
    sentiment labels with confidence scores.

11. What are the three sentiment classes in this assignment?
    Negative, Neutral and Positive.

12. What does LABEL_0 represent in this model?
    Negative sentiment.

13. What does LABEL_1 represent in this model?
    Neutral sentiment.

14. What does LABEL_2 represent in this model?
    Positive sentiment.

15. What is a confidence score?
    It is the model's score associated with its predicted class. It should
    not automatically be treated as a perfectly calibrated probability.

16. What is tokenization?
    Tokenization divides text into smaller units called tokens.

17. What is attention in Transformers?
    Attention helps the model determine which other tokens are important
    when processing a particular token.

18. What is self-attention?
    Self-attention calculates relationships between tokens within the
    same input sequence.

19. What is the advantage of Transformer models over simple rule-based methods?
    Transformers can learn contextual language patterns and handle more
    complex expressions than simple keyword rules.

20. Why is a multilingual model useful?
    It can process text written in multiple languages using one model.

21. What is a CSV file?
    CSV stands for Comma-Separated Values. It stores tabular data as text.

22. Why are text and lang columns used?
    The text column contains the tweet or opinion, while lang identifies
    its language code for reference in the output.

23. What is batch processing?
    Batch processing sends multiple text inputs for processing together,
    rather than making a separate model call for every row.

24. What does truncation=True do?
    It truncates inputs that exceed the model's supported maximum length.

25. Why is internet access needed?
    The pretrained model files may need to be downloaded from Hugging Face
    the first time the program runs.

26. Does the program train the Transformer model?
    No. It loads an already pretrained model and uses it for prediction.

27. What is inference?
    Inference is the process of using a trained model to make predictions
    on new input data.

28. What is the difference between training and inference?
    Training updates model parameters using data. Inference uses the
    existing model parameters to generate predictions.

29. What is the purpose of the sentiment summary?
    It counts how many tweets were classified as negative, neutral and positive.

30. What are limitations of this assignment?
    Predictions may be affected by sarcasm, slang, spelling errors,
    domain-specific language and mixed-language text. The model's output
    should be reviewed before making important decisions.

NOTE:
This assignment performs inference using a pretrained model. It does not
fine-tune the model on the CSV dataset. Model outputs are predictions and
may not always reflect the true meaning of a tweet.
"""
