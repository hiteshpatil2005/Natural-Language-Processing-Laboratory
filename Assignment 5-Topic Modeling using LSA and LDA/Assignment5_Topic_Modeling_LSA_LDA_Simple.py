import csv
import math
import os
import random
import re
from collections import Counter


# --------------------------- SAMPLE TELECOM DATA ---------------------------

SAMPLE_TICKETS = [
    "Internet connection is very slow and broadband speed keeps dropping",
    "WiFi router disconnects frequently and internet is not working",
    "Broadband internet is down and the modem has no signal",
    "Unable to connect to WiFi network after router restart",
    "Mobile data internet speed is slow and network signal is weak",
    "Phone call drops during conversation and voice quality is poor",
    "Outgoing calls are not connecting and incoming calls fail",
    "Voice calls have disturbance and network coverage is poor",
    "SIM card is not registered and mobile network is unavailable",
    "Cannot make calls because the mobile signal keeps disappearing",
    "My monthly bill contains an incorrect charge",
    "I was charged twice for the same recharge plan",
    "Need refund for failed online payment and recharge",
    "Payment was deducted but my prepaid recharge was not activated",
    "The postpaid bill amount is higher than the selected plan",
    "Please explain extra data charges on my monthly bill",
    "Customer service has not resolved my broadband complaint",
    "I cannot reach customer care and need technical support",
    "Technician visit was scheduled but nobody arrived",
    "Complaint status is not updated after service request",
    "Need to change my mobile plan and data package",
    "Please activate the new unlimited calling plan",
    "My data pack expired before the validity date",
    "Unable to activate roaming service while travelling",
]


# ------------------------------- PREPROCESSING ------------------------------

STOP_WORDS = {
    "the", "is", "a", "an", "and", "or", "to", "for", "of", "in", "on",
    "my", "i", "it", "was", "are", "am", "be", "been", "has", "have",
    "had", "with", "after", "before", "during", "this", "that", "please",
    "not", "but", "very", "keeps", "same", "need", "unable", "cannot",
    "no", "your", "you", "me", "we", "our", "they", "their", "from",
}


def tokenize(text):
    """Convert text into lowercase words and remove common stop words."""
    words = re.findall(r"[a-z]+", text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 1]


def load_documents():
    """Read telecom_support_tickets.csv or use the built-in sample tickets."""
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "telecom_support_tickets.csv"
    )

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames and "text" in reader.fieldnames:
                    documents = [
                        row.get("text", "").strip()
                        for row in reader
                        if row.get("text", "").strip()
                    ]
                    if documents:
                        print("Dataset loaded from telecom_support_tickets.csv")
                        return documents
                print("CSV has no usable 'text' column. Using sample data.")
        except (OSError, csv.Error) as error:
            print("Could not read CSV:", error)
            print("Using built-in sample data.")
    else:
        print("CSV not found. Using built-in sample telecom tickets.")

    return SAMPLE_TICKETS[:]


# ---------------------------- VECTOR CREATION -------------------------------

def build_vocabulary(tokenized_documents):
    """Create a sorted vocabulary and word-to-index dictionary."""
    vocabulary = sorted({
        word
        for document in tokenized_documents
        for word in document
    })
    word_to_id = {word: index for index, word in enumerate(vocabulary)}
    return vocabulary, word_to_id


def create_tf_idf(tokenized_documents, vocabulary, word_to_id):
    """Create a TF-IDF matrix using lists (no NumPy required)."""
    document_count = len(tokenized_documents)
    vocabulary_size = len(vocabulary)

    document_frequency = [0] * vocabulary_size
    for document in tokenized_documents:
        for word in set(document):
            document_frequency[word_to_id[word]] += 1

    matrix = []
    for document in tokenized_documents:
        counts = Counter(document)
        total_words = max(1, len(document))
        row = []

        for word in vocabulary:
            word_id = word_to_id[word]
            term_frequency = counts[word] / total_words
            inverse_document_frequency = math.log(
                (1 + document_count) / (1 + document_frequency[word_id])
            ) + 1
            row.append(term_frequency * inverse_document_frequency)

        matrix.append(row)

    return matrix


def normalize(vector):
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        return vector[:]
    return [value / length for value in vector]


def dot(vector_a, vector_b):
    return sum(a * b for a, b in zip(vector_a, vector_b))


# ---------------------------- LSA USING SVD --------------------------------

def multiply_matrix_vector(matrix, vector):
    return [dot(row, vector) for row in matrix]


def transpose(matrix):
    if not matrix:
        return []
    return [list(column) for column in zip(*matrix)]


def lsa_topics(tfidf_matrix, vocabulary, number_of_topics=3, top_words=6):
    """
    Approximate truncated SVD using power iteration on A-transpose times A.

    Each resulting component is used to identify the strongest topic words.
    """
    if not tfidf_matrix or not vocabulary:
        return []

    document_count = len(tfidf_matrix)
    vocabulary_size = len(vocabulary)
    actual_topics = min(number_of_topics, vocabulary_size, document_count)

    if actual_topics < 1:
        return []

    matrix_t = transpose(tfidf_matrix)
    gram_matrix = [
        [
            sum(matrix_t[i][doc] * tfidf_matrix[doc][j]
                for doc in range(document_count))
            for j in range(vocabulary_size)
        ]
        for i in range(vocabulary_size)
    ]

    components = []
    random.seed(42)

    for _ in range(actual_topics):
        vector = normalize([random.random() + 0.1
                            for _ in range(vocabulary_size)])

        # Power iteration approximates a dominant eigenvector.
        for _iteration in range(80):
            next_vector = normalize(
                multiply_matrix_vector(gram_matrix, vector)
            )
            if not any(next_vector):
                break
            vector = next_vector

        components.append(vector)

        # Deflation: remove this component's contribution before finding
        # the next component.
        eigenvalue = dot(vector, multiply_matrix_vector(gram_matrix, vector))
        for i in range(vocabulary_size):
            for j in range(vocabulary_size):
                gram_matrix[i][j] -= (
                    eigenvalue * vector[i] * vector[j]
                )

    results = []
    for topic_number, component in enumerate(components, start=1):
        ranked = sorted(
            range(vocabulary_size),
            key=lambda index: abs(component[index]),
            reverse=True
        )
        keywords = [vocabulary[index] for index in ranked[:top_words]]
        results.append(("LSA Topic " + str(topic_number), keywords))

    return results


# ---------------------------- LDA USING GIBBS ------------------------------

def lda_topics(tokenized_documents, vocabulary, word_to_id,
               number_of_topics=3, top_words=6, iterations=100):
    """
    Simplified LDA using collapsed Gibbs sampling.

    Each document contains topic assignments for its words. During training,
    a topic is sampled using document-topic and topic-word probabilities.
    """
    random.seed(42)

    document_count = len(tokenized_documents)
    vocabulary_size = len(vocabulary)
    actual_topics = min(number_of_topics, vocabulary_size)

    if document_count == 0 or vocabulary_size == 0 or actual_topics == 0:
        return [], []

    alpha = 0.1
    beta = 0.01

    document_topic_counts = [
        [0] * actual_topics for _ in range(document_count)
    ]
    topic_word_counts = [
        [0] * vocabulary_size for _ in range(actual_topics)
    ]
    topic_totals = [0] * actual_topics
    assignments = []

    # Randomly assign each word token to a topic.
    for doc_id, document in enumerate(tokenized_documents):
        document_assignments = []

        for word in document:
            word_id = word_to_id[word]
            topic = random.randrange(actual_topics)

            document_assignments.append(topic)
            document_topic_counts[doc_id][topic] += 1
            topic_word_counts[topic][word_id] += 1
            topic_totals[topic] += 1

        assignments.append(document_assignments)

    # Gibbs sampling iterations.
    for _ in range(iterations):
        for doc_id, document in enumerate(tokenized_documents):
            for position, word in enumerate(document):
                word_id = word_to_id[word]
                old_topic = assignments[doc_id][position]

                # Remove the current token from its old topic.
                document_topic_counts[doc_id][old_topic] -= 1
                topic_word_counts[old_topic][word_id] -= 1
                topic_totals[old_topic] -= 1

                probabilities = []
                for topic in range(actual_topics):
                    document_probability = (
                        document_topic_counts[doc_id][topic] + alpha
                    ) / (
                        len(document) - 1 + actual_topics * alpha
                    )

                    word_probability = (
                        topic_word_counts[topic][word_id] + beta
                    ) / (
                        topic_totals[topic] + vocabulary_size * beta
                    )

                    probabilities.append(
                        document_probability * word_probability
                    )

                total_probability = sum(probabilities)
                if total_probability == 0:
                    new_topic = random.randrange(actual_topics)
                else:
                    pick = random.random() * total_probability
                    cumulative = 0.0
                    new_topic = actual_topics - 1

                    for topic, probability in enumerate(probabilities):
                        cumulative += probability
                        if pick <= cumulative:
                            new_topic = topic
                            break

                # Add the token to the newly selected topic.
                assignments[doc_id][position] = new_topic
                document_topic_counts[doc_id][new_topic] += 1
                topic_word_counts[new_topic][word_id] += 1
                topic_totals[new_topic] += 1

    results = []
    for topic in range(actual_topics):
        ranked = sorted(
            range(vocabulary_size),
            key=lambda word_id: topic_word_counts[topic][word_id],
            reverse=True
        )
        keywords = [
            vocabulary[word_id]
            for word_id in ranked[:top_words]
            if topic_word_counts[topic][word_id] > 0
        ]
        results.append(("LDA Topic " + str(topic + 1), keywords))

    return results, document_topic_counts


# ----------------------------- TOPIC COHERENCE ------------------------------

def calculate_coherence(topics, tokenized_documents):
    """
    Calculate a simple UMass-style coherence score for each topic.

    Higher (less negative) scores generally indicate stronger co-occurrence
    among a topic's top words. This is a basic educational implementation.
    """
    document_sets = [set(document) for document in tokenized_documents]
    document_count = len(document_sets)
    scores = {}

    for topic_name, words in topics:
        pair_scores = []

        for i in range(1, len(words)):
            word_i = words[i]

            for j in range(i):
                word_j = words[j]

                count_j = sum(word_j in doc for doc in document_sets)
                count_both = sum(
                    word_i in doc and word_j in doc
                    for doc in document_sets
                )

                if count_j > 0:
                    pair_scores.append(
                        math.log((count_both + 1) / count_j)
                    )

        scores[topic_name] = (
            sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
        )

    return scores


# --------------------------- DISPLAY / ANALYSIS -----------------------------

def display_topics(title, topics, coherence):
    print("\n" + title)
    print("-" * len(title))

    if not topics:
        print("No topics could be generated.")
        return

    for topic_name, keywords in topics:
        score = coherence.get(topic_name, 0.0)
        print("{}: {}".format(topic_name, ", ".join(keywords)))
        print("  Coherence score: {:.4f}".format(score))


def main():
    print("=" * 65)
    print("ASSIGNMENT 5: TOPIC MODELING USING LSA / LDA")
    print("=" * 65)

    documents = load_documents()
    tokenized_documents = [
        tokenize(document) for document in documents
    ]
    tokenized_documents = [
        document for document in tokenized_documents if document
    ]

    if not tokenized_documents:
        print("No valid text documents found.")
        return

    print("\nLoaded {} telecom support tickets.".format(
        len(tokenized_documents)
    ))

    vocabulary, word_to_id = build_vocabulary(tokenized_documents)
    print("Vocabulary size:", len(vocabulary))

    # Create TF-IDF features for LSA.
    tfidf_matrix = create_tf_idf(
        tokenized_documents, vocabulary, word_to_id
    )

    # Run LSA.
    lsa_results = lsa_topics(
        tfidf_matrix, vocabulary, number_of_topics=3
    )
    lsa_coherence = calculate_coherence(
        lsa_results, tokenized_documents
    )
    display_topics(
        "TOPICS DISCOVERED USING LSA (TF-IDF + SVD)",
        lsa_results,
        lsa_coherence
    )

    # Run LDA.
    lda_results, document_topic_counts = lda_topics(
        tokenized_documents,
        vocabulary,
        word_to_id,
        number_of_topics=3
    )
    lda_coherence = calculate_coherence(
        lda_results, tokenized_documents
    )
    display_topics(
        "TOPICS DISCOVERED USING LDA (GIBBS SAMPLING)",
        lda_results,
        lda_coherence
    )

    # Show the dominant LDA topic for each ticket.
    print("\nTICKET TOPIC ASSIGNMENTS")
    print("------------------------")
    for index, counts in enumerate(document_topic_counts):
        dominant_topic = counts.index(max(counts)) + 1
        preview = documents[index][:65]
        print("Ticket {:02d} -> Topic {} | {}".format(
            index + 1, dominant_topic, preview
        ))

    print("\nOPERATIONAL INSIGHT")
    print("-------------------")
    print("Review the keywords in each topic to identify common customer")
    print("problems, such as internet connectivity, call quality, billing,")
    print("recharge, or customer support. Use these patterns to prioritize")
    print("network maintenance and customer-service improvements.")

    print("\nAssignment 5 completed successfully.")


if __name__ == "__main__":
    main()












































"""
ASSIGNMENT 5: TOPIC MODELING USING LSA / LDA
================================================
Simple version using Python standard library only.

Topics covered:
1. Text preprocessing and tokenization
2. TF-IDF calculation
3. LSA (Latent Semantic Analysis) using a simple power-iteration SVD method
4. LDA (Latent Dirichlet Allocation) using collapsed Gibbs sampling
5. Topic keyword extraction
6. Topic coherence (UMass-style coherence)
7. Telecom support ticket analysis

No external packages are required. No internet connection is required.
The program uses built-in sample telecom tickets if the CSV file is absent.
"""

# ---------------------------------------------------------------------------
# INSTALLATION AND EXECUTION
# ---------------------------------------------------------------------------
# Requirements: Python 3.8 or later. No pip installation is required.
#
# WINDOWS (PowerShell):
#   1. Open PowerShell in the folder containing this file.
#   2. Run:
#        python Assignment5_Topic_Modeling_LSA_LDA_Simple.py
#      Or:
#        py Assignment5_Topic_Modeling_LSA_LDA_Simple.py
#
# UBUNTU / LINUX:
#   1. Install Python if required:
#        sudo apt update
#        sudo apt install python3
#   2. Run:
#        python3 Assignment5_Topic_Modeling_LSA_LDA_Simple.py
#
# OPTIONAL CSV INPUT:
#   Place telecom_support_tickets.csv in the same folder as this script.
#   The CSV must contain a column named "text".
#   If the file is missing or contains no usable rows, built-in sample tickets
#   are used automatically.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# VIVA / ORAL QUESTIONS AND ANSWERS
# ---------------------------------------------------------------------------
#
# Q1. What is topic modeling?
# Ans: Topic modeling is an unsupervised technique used to discover hidden
#      themes or topics in a collection of text documents.
#
# Q2. What is LSA?
# Ans: Latent Semantic Analysis discovers relationships between documents
#      and terms using a term-document matrix and matrix factorization.
#
# Q3. What is SVD?
# Ans: Singular Value Decomposition is a matrix factorization technique that
#      represents a matrix using three matrices: U, Sigma, and V-transpose.
#
# Q4. What is the role of SVD in LSA?
# Ans: SVD reduces the dimensions of the term-document representation and
#      helps identify latent semantic patterns.
#
# Q5. What is LDA?
# Ans: Latent Dirichlet Allocation is a probabilistic topic model that
#      represents documents as mixtures of topics.
#
# Q6. Is LDA supervised or unsupervised?
# Ans: LDA topic modeling is an unsupervised learning technique.
#
# Q7. What is a topic in LDA?
# Ans: A topic is a probability distribution over words.
#
# Q8. What is a document-topic distribution?
# Ans: It represents the proportion or probability of different topics in
#      a document.
#
# Q9. What is a topic-word distribution?
# Ans: It represents the probability of words belonging to a topic.
#
# Q10. What is a Dirichlet distribution?
# Ans: It is a probability distribution used as a prior for topic mixtures
#       and word distributions in LDA.
#
# Q11. What is Gibbs sampling?
# Ans: Gibbs sampling is a sampling method that repeatedly updates one
#       variable using the current values of the other variables.
#
# Q12. What is a corpus?
# Ans: A corpus is a collection of text documents used for analysis.
#
# Q13. What is tokenization?
# Ans: Tokenization splits text into smaller units, usually words.
#
# Q14. What are stop words?
# Ans: Stop words are common words that are often removed because they
#       contribute little to topic discovery.
#
# Q15. What is TF-IDF?
# Ans: TF-IDF measures the importance of a word in a document relative to
#       its frequency across the document collection.
#
# Q16. Why is TF-IDF used in LSA?
# Ans: It gives more weight to informative words and less weight to words
#       that occur in many documents.
#
# Q17. What is a document-term matrix?
# Ans: It is a matrix where rows represent documents and columns represent
#       terms, with values showing term counts or weights.
#
# Q18. What is topic keyword extraction?
# Ans: It selects the words with the highest topic weights or probabilities
#       to describe each topic.
#
# Q19. What is topic coherence?
# Ans: Topic coherence estimates how semantically related the top words
#       within a topic are.
#
# Q20. What does a higher coherence score generally indicate?
# Ans: It generally indicates that the topic's important words co-occur
#       more consistently.
#
# Q21. What is the difference between LSA and LDA?
# Ans: LSA is based on linear algebra and matrix factorization, whereas
#       LDA is a probabilistic generative topic model.
#
# Q22. What is the purpose of the number of topics?
# Ans: It specifies how many topic groups the model should discover.
#
# Q23. What is alpha in LDA?
# Ans: Alpha controls the document-topic distribution prior.
#
# Q24. What is beta in LDA?
# Ans: Beta controls the topic-word distribution prior.
#
# Q25. What is an iteration in Gibbs sampling?
# Ans: One iteration updates topic assignments for the word tokens in
#       the documents.
#
# Q26. Why is topic modeling useful for telecom support tickets?
# Ans: It can group common complaints such as network issues, billing
#       problems, recharge failures, and customer support requests.
#
# Q27. How can telecom companies use the discovered topics?
# Ans: They can identify frequent issues, prioritize maintenance, and
#       improve customer support operations.
#
# Q28. What is the limitation of LSA?
# Ans: LSA is a linear method and may not capture all complex semantic
#       relationships in text.
#
# Q29. What is the limitation of LDA?
# Ans: The number of topics must be selected, and results can depend on
#       preprocessing and model parameters.
#
# Q30. Does this simplified program require scikit-learn?
# Ans: No. It uses Python's standard library and implements educational
#       versions of the topic-modeling methods.
#
# IMPORTANT NOTE:
# This program uses simplified educational implementations of SVD-based
# LSA and Gibbs-sampling LDA. Results may differ from scikit-learn or other
# optimized topic-modeling libraries, especially on large datasets.
# ---------------------------------------------------------------------------
