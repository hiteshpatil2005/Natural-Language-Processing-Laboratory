'''
Name: Patil Hitesh Dattatraya
Class: Btech IT 
Batch: B2
Roll:  106
PRN:   UIT23M1106
Assignment 8: Transformer-based Sentiment Analysis using Hugging Face
'''
import os
import csv
from transformers import pipeline


def predict_sentiment(texts):
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    classifier = pipeline("sentiment-analysis", model=model_name)

    results = []

    for text in texts:
        prediction = classifier(text)[0]

        label = prediction["label"]

        if label == "LABEL_0":
            label = "Negative"
        elif label == "LABEL_1":
            label = "Neutral"
        else:
            label = "Positive"

        results.append({
            "text": text,
            "label": label,
            "score": round(prediction["score"], 4)
        })

    return results


def main():
    print("=" * 75)
    print("MULTILINGUAL SENTIMENT ANALYSIS")
    print("=" * 75)

    file_path = os.path.join(os.path.dirname(__file__),"multilingual_opinion_tweets.csv")

    if not os.path.exists(file_path):
        print("CSV file not found:", file_path)
        return

    tweets = []
    languages = []

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            tweets.append(row["text"])
            languages.append(row["lang"].upper())

    try:
        results = predict_sentiment(tweets)

        print(f"\n{'Language':<10} {'Sentiment':<12} {'Confidence':<12} Tweet")
        print("-" * 85)

        for language, result in zip(languages, results):
            print(
                f"{language:<10} "
                f"{result['label']:<12} "
                f"{result['score']:<12} "
                f"{result['text']}"
            )

    except Exception as error:
        print("Error loading the sentiment model:", error)
        print("Please check your internet connection and installed libraries.")


if __name__ == "__main__":
    main()
