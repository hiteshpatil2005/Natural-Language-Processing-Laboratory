"""
Assignment 3: Bag-of-Words & TF-IDF Vectorization
- Bag-of-Words (BoW) & TF-IDF vector representations built from scratch using NumPy
- Verification with scikit-learn CountVectorizer & TfidfVectorizer
- Document-similarity & Plagiarism Detection system based on Cosine Similarity
"""

import os
import json
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def bow_from_scratch(corpus: List[str]) -> Tuple[List[str], np.ndarray]:
    """Generates vocabulary and Bag-of-Words matrix from scratch."""
    vocab = sorted(list(set(w.lower() for doc in corpus for w in doc.split())))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    
    matrix = np.zeros((len(corpus), len(vocab)), dtype=int)
    for row_idx, doc in enumerate(corpus):
        for word in doc.split():
            w_clean = word.lower()
            if w_clean in word_to_idx:
                matrix[row_idx, word_to_idx[w_clean]] += 1
    return vocab, matrix

def tfidf_from_scratch(corpus: List[str]) -> Tuple[List[str], np.ndarray]:
    """Generates normalized TF-IDF feature matrix from scratch."""
    vocab, bow = bow_from_scratch(corpus)
    N = len(corpus)
    
    # Term Frequency (TF)
    doc_lengths = bow.sum(axis=1, keepdims=True)
    tf = bow / np.maximum(doc_lengths, 1)
    
    # Inverse Document Frequency (IDF)
    df = (bow > 0).sum(axis=0)
    idf = np.log((N + 1) / (df + 1)) + 1
    
    tfidf = tf * idf
    
    # L2 Normalization
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    tfidf_normalized = np.where(norms > 0, tfidf / norms, tfidf)
    return vocab, tfidf_normalized

def run_plagiarism_check(corpus: List[str], doc_names: List[str], threshold: float = 0.70):
    """Calculates cosine similarity across document pairs to flag potential plagiarism."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus).toarray()
    sim_matrix = cosine_similarity(tfidf_matrix)

    print(f"{'Document A':<20} | {'Document B':<20} | {'Cosine Sim':<12} | {'Status':<15}")
    print("-" * 75)

    for i in range(len(doc_names)):
        for j in range(i + 1, len(doc_names)):
            score = sim_matrix[i, j]
            status = "FLAGGED PLAGIARISM" if score >= threshold else "CLEAN"
            print(f"{doc_names[i]:<20} | {doc_names[j]:<20} | {score:.4f}       | {status:<15}")

def main():
    print("=" * 60)
    print("ASSIGNMENT 3: BAG-OF-WORDS & TF-IDF VECTORIZATION")
    print("=" * 60)

    data_path = os.path.join(os.path.dirname(__file__),"academic_submissions.json")
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus = [item["content"] for item in data]
    doc_ids = [item["id"] for item in data]

    # 1. From-Scratch Verification
    vocab_scratch, tfidf_scratch = tfidf_from_scratch(corpus)
    print(f"\nVocabulary Size (From Scratch): {len(vocab_scratch)} words")
    print(f"TF-IDF Matrix Shape: {tfidf_scratch.shape}")

    # 2. Scikit-learn Comparison
    sklearn_vec = TfidfVectorizer()
    tfidf_sklearn = sklearn_vec.fit_transform(corpus).toarray()
    print(f"Scikit-Learn TF-IDF Matrix Shape: {tfidf_sklearn.shape}")

    # 3. Academic Plagiarism Detection Module
    print("\n--- Academic Integrity Plagiarism Audit Report ---")
    run_plagiarism_check(corpus, doc_ids, threshold=0.70)

if __name__ == "__main__":
    main()
