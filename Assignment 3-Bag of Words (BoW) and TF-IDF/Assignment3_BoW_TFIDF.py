import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Create a sample dataset if academic_submissions.json is not available
sample_data = [
    {
        "id": "DOC1",
        "class": "Technology",
        "content": "Artificial intelligence and machine learning are used in modern technology."
    },
    {
        "id": "DOC2",
        "class": "Technology",
        "content": "Machine learning and artificial intelligence help in data analysis."
    },
    {
        "id": "DOC3",
        "class": "Environment",
        "content": "Environmental protection includes clean energy and reducing pollution."
    },
    {
        "id": "DOC4",
        "class": "Environment",
        "content": "Clean energy helps reduce pollution and protect the environment."
    }
]

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "academic_submissions.json")

if not os.path.exists(data_path):
    with open(data_path, "w", encoding="utf-8") as file:
        json.dump(sample_data, file, indent=4)
    print("Sample academic_submissions.json created.")


# 2. Load dataset
with open(data_path, "r", encoding="utf-8") as file:
    data = json.load(file)

corpus = [item["content"] for item in data]
doc_ids = [item.get("id", f"Document{i+1}") for i, item in enumerate(data)]
classes = [item.get("class", "General") for item in data]

print("\n========== DATASET ==========")
print("Number of documents:", len(corpus))
for doc_id, text in zip(doc_ids, corpus):
    print(f"\n{doc_id}: {text}")


# 3. Bag-of-Words from scratch
def get_words(text):
    # Use words with at least two characters, similar to CountVectorizer defaults.
    return re.findall(r"(?u)\b\w\w+\b", text.lower())


def bow_from_scratch(documents):
    vocabulary = sorted(set(word for doc in documents for word in get_words(doc)))
    word_index = {word: index for index, word in enumerate(vocabulary)}

    bow_matrix = np.zeros((len(documents), len(vocabulary)), dtype=int)

    for row, document in enumerate(documents):
        for word in get_words(document):
            bow_matrix[row, word_index[word]] += 1

    return vocabulary, bow_matrix


vocabulary, bow_matrix = bow_from_scratch(corpus)

print("\n========== BAG-OF-WORDS FROM SCRATCH ==========")
print("Vocabulary:", vocabulary)
print("BoW Matrix:\n", bow_matrix)


# 4. Verify Bag-of-Words using scikit-learn CountVectorizer
count_vectorizer = CountVectorizer()
bow_sklearn = count_vectorizer.fit_transform(corpus).toarray()

print("\n========== SCIKIT-LEARN COUNT VECTORIZER ==========")
print("Vocabulary:", count_vectorizer.get_feature_names_out())
print("BoW Matrix:\n", bow_sklearn)


# 5. TF-IDF from scratch
def tfidf_from_scratch(bow):
    total_documents = bow.shape[0]

    # Number of documents containing each word
    document_frequency = np.count_nonzero(bow, axis=0)

    # Smoothed Inverse Document Frequency
    idf = np.log((1 + total_documents) / (1 + document_frequency)) + 1

    # TF-IDF and L2 normalization
    tfidf = bow * idf
    row_norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    tfidf = np.divide(tfidf, row_norms, out=np.zeros_like(tfidf, dtype=float),
                      where=row_norms != 0)

    return tfidf


tfidf_scratch = tfidf_from_scratch(bow_matrix)

print("\n========== TF-IDF FROM SCRATCH ==========")
print(np.round(tfidf_scratch, 3))


# 6. Verify TF-IDF using scikit-learn TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_sklearn = tfidf_vectorizer.fit_transform(corpus).toarray()

print("\n========== SCIKIT-LEARN TF-IDF ==========")
print(np.round(tfidf_sklearn, 3))

print("\nScratch and scikit-learn matrix shapes:",
      tfidf_scratch.shape, tfidf_sklearn.shape)


# 7. Document similarity using cosine similarity
similarity_matrix = cosine_similarity(tfidf_sklearn)

print("\n========== DOCUMENT SIMILARITY MATRIX ==========")
print(np.round(similarity_matrix, 3))


# 8. Plagiarism detection (syllabus threshold: 0.85)
threshold = 0.85

print("\n========== PLAGIARISM DETECTION ==========")
print(f"{'Document A':<15} {'Document B':<15} {'Similarity':<12} Status")
print("-" * 65)

for i in range(len(doc_ids)):
    for j in range(i + 1, len(doc_ids)):
        score = similarity_matrix[i, j]
        status = "POSSIBLE PLAGIARISM" if score > threshold else "Not flagged"
        print(f"{doc_ids[i]:<15} {doc_ids[j]:<15} {score:<12.3f} {status}")


# 9. Visualize top keywords for each document class
print("\n========== TOP KEYWORDS BY CLASS ==========")

for category in sorted(set(classes)):
    selected_docs = [corpus[i] for i in range(len(corpus))
                     if classes[i] == category]

    class_vectorizer = TfidfVectorizer()
    class_matrix = class_vectorizer.fit_transform(selected_docs)

    scores = np.asarray(class_matrix.sum(axis=0)).ravel()
    words = class_vectorizer.get_feature_names_out()

    top_indices = scores.argsort()[-5:][::-1]
    top_words = words[top_indices]
    top_scores = scores[top_indices]

    print(f"\n{category}:")
    for word, score in zip(top_words, top_scores):
        print(f"{word}: {score:.3f}")

    plt.figure(figsize=(7, 4))
    plt.bar(top_words, top_scores)
    plt.title("Top Keywords - " + category)
    plt.xlabel("Keywords")
    plt.ylabel("TF-IDF Score")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("top_keywords_" + re.sub(r"\W+", "_", category) + ".png")
    plt.show()

print("\nAssignment 3 completed successfully.")
print("Keyword charts have been saved in the current folder.")
































"""
ASSIGNMENT 3: BAG-OF-WORDS & TF-IDF VECTORIZATION
PCIT404 - Natural Language Processing Laboratory

============================================================
INSTALLATION / DEPENDENCY COMMANDS
============================================================

A) WINDOWS (Command Prompt or PowerShell)
-----------------------------------------
1. Create and activate a virtual environment:

   py -3.11 -m venv venv
   venv\\Scripts\\activate

2. Install required libraries:

   python -m pip install --upgrade pip
   pip install numpy scikit-learn matplotlib

3. Run the program:

   python Assignment3_BoW_TFIDF.py


B) UBUNTU / LINUX (Terminal)
----------------------------
1. Install Python and virtual-environment support:

   sudo apt update
   sudo apt install python3 python3-venv python3-pip

2. Create and activate a virtual environment:

   python3 -m venv venv
   source venv/bin/activate

3. Install required libraries:

   python -m pip install --upgrade pip
   pip install numpy scikit-learn matplotlib

4. Run the program:

   python Assignment3_BoW_TFIDF.py


NOTES:
- Python 3.10 or 3.11 is recommended.
- academic_submissions.json should be in the same folder as this file.
- If the JSON dataset is missing, this program creates a small sample
  dataset automatically so that you can run and understand the practical.
- The program saves a top-keywords bar chart as top_keywords.png.
============================================================
"""

# ============================================================
# VIVA / ORAL EXAMINATION THEORY
# ============================================================

"""
1. What is text vectorization?
   Text vectorization converts text documents into numerical values
   so that machine-learning algorithms can process them.

2. What is Bag-of-Words (BoW)?
   Bag-of-Words represents a document by counting how many times
   each word appears. It does not consider word order.

3. What is vocabulary?
   Vocabulary is the collection of all unique words found in the
   document collection.

4. What is a BoW matrix?
   It is a matrix in which rows represent documents and columns
   represent vocabulary words. Each value represents a word count.

5. What is NumPy?
   NumPy is a Python library used for numerical calculations and
   working with arrays and matrices.

6. What is TF-IDF?
   TF-IDF stands for Term Frequency-Inverse Document Frequency.
   It gives importance to a word based on its frequency in a document
   and how rare it is across all documents.

7. What is Term Frequency (TF)?
   TF measures how often a word appears in a document.

8. What is Document Frequency (DF)?
   DF is the number of documents in which a particular word appears.

9. What is Inverse Document Frequency (IDF)?
   IDF gives higher importance to words that occur in fewer documents.
   Common words across many documents receive lower importance.

10. Write the IDF formula used in this practical.
    IDF = log((1 + N) / (1 + DF)) + 1
    where N is the total number of documents and DF is the number
    of documents containing the word.

11. What is the difference between BoW and TF-IDF?
    BoW stores word counts. TF-IDF gives weighted values based on
    word frequency and rarity across the document collection.

12. What is CountVectorizer?
    CountVectorizer is a scikit-learn class that converts text
    documents into a matrix of word counts.

13. What is TfidfVectorizer?
    TfidfVectorizer converts text documents into TF-IDF features.

14. Why do we implement BoW and TF-IDF from scratch?
    To understand how vocabulary, word counts, TF and IDF are
    calculated internally.

15. Why do we verify results using scikit-learn?
    It helps compare our implementation with standard library tools.

16. What is cosine similarity?
    Cosine similarity measures how similar two vectors are by
    calculating the cosine of the angle between them.

17. What is the range of cosine similarity?
    For non-negative TF-IDF vectors, it ranges from 0 to 1.
    1 means identical direction and 0 means no shared weighted terms.

18. What is plagiarism detection?
    It compares documents and identifies pairs with high textual
    similarity for further review.

19. What threshold is used in this assignment?
    0.85. A pair with cosine similarity greater than 0.85 is flagged
    as possible plagiarism.

20. Does high cosine similarity prove plagiarism?
    No. It only indicates textual similarity. Human review is needed
    before deciding whether plagiarism has occurred.

21. What is a sparse matrix?
    A matrix in which most values are zero. Text vectorization
    commonly produces sparse matrices because each document uses
    only a small part of the complete vocabulary.

22. What is L2 normalization?
    It scales a vector so that its Euclidean length becomes 1,
    provided the vector is non-zero.

23. What is a document corpus?
    A corpus is a collection of documents used for text analysis.

24. What is the purpose of a bar chart in this assignment?
    It displays the top keywords and their TF-IDF scores for each
    document class.

25. Which libraries are used?
    NumPy: numerical arrays and matrix operations.
    scikit-learn: CountVectorizer, TfidfVectorizer and cosine similarity.
    Matplotlib: keyword bar charts.
    json: reading the academic submissions dataset.

26. What is the output of this practical?
    BoW matrix, TF-IDF matrix, cosine similarity matrix, plagiarism
    report and top-keyword bar charts.

27. What are the limitations of BoW?
    It ignores word order and context, and the vocabulary can become
    very large.

28. What are the limitations of TF-IDF?
    It does not understand word meaning or context. Similar meanings
    expressed using different words may not be recognized.

29. What is the difference between CountVectorizer and TfidfVectorizer?
    CountVectorizer returns word counts, whereas TfidfVectorizer
    returns weighted TF-IDF values.

30. What is the main objective of this assignment?
    To implement BoW and TF-IDF from scratch, verify them using
    scikit-learn, calculate document similarity and identify
    potentially similar academic submissions.

============================================================
IMPORTANT FUNCTIONS TO REMEMBER
============================================================
np.zeros()                       - create an array of zeros
np.count_nonzero()               - count non-zero values
np.log()                         - calculate natural logarithm
np.linalg.norm()                 - calculate vector length
CountVectorizer()                - convert text to word counts
TfidfVectorizer()                - convert text to TF-IDF
cosine_similarity()              - calculate document similarity
json.load()                       - read JSON data
json.dump()                       - write JSON data
plt.bar()                         - create a bar chart
plt.savefig()                     - save a chart
plt.show()                        - display a chart
============================================================
"""
