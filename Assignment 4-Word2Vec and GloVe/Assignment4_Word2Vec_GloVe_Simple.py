import math
import random
import re
from collections import Counter, defaultdict


# ---------------------------- SAMPLE DATA ----------------------------------

PRODUCTS = [
    {
        "name": "smartphone",
        "description": "smartphone mobile phone camera screen battery android"
    },
    {
        "name": "android phone",
        "description": "android smartphone mobile phone battery camera"
    },
    {
        "name": "laptop",
        "description": "laptop computer keyboard processor memory screen"
    },
    {
        "name": "gaming laptop",
        "description": "gaming laptop computer processor graphics keyboard"
    },
    {
        "name": "headphones",
        "description": "headphones audio music sound wireless bluetooth"
    },
    {
        "name": "wireless earbuds",
        "description": "wireless earbuds bluetooth headphones audio music"
    },
    {
        "name": "smart watch",
        "description": "smart watch fitness health heart monitor bluetooth"
    },
    {
        "name": "fitness band",
        "description": "fitness band health step monitor smart watch"
    },
]

# A few additional sentences help the models learn word relationships.
EXTRA_SENTENCES = [
    "mobile phone has camera screen and battery",
    "android phone is a smartphone",
    "laptop computer has keyboard processor and memory",
    "gaming laptop has graphics processor",
    "headphones play music and audio",
    "wireless earbuds use bluetooth for music",
    "smart watch tracks fitness and health",
    "fitness band tracks steps and heart health",
    "phone and laptop are electronic devices",
    "bluetooth connects wireless headphones and earbuds",
]


# ---------------------------- PREPROCESSING --------------------------------

def tokenize(text):
    """Convert text to lowercase words and remove punctuation."""
    return re.findall(r"[a-z]+", text.lower())


def prepare_corpus():
    """Create tokenized sentences from the sample products and extra text."""
    sentences = []

    for product in PRODUCTS:
        sentences.append(tokenize(product["description"]))

    for sentence in EXTRA_SENTENCES:
        sentences.append(tokenize(sentence))

    return [sentence for sentence in sentences if sentence]


# ---------------------------- COMMON HELPERS -------------------------------

def dot(vector_a, vector_b):
    return sum(a * b for a, b in zip(vector_a, vector_b))


def vector_norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(vector_a, vector_b):
    denominator = vector_norm(vector_a) * vector_norm(vector_b)
    if denominator == 0:
        return 0.0
    return dot(vector_a, vector_b) / denominator


def sigmoid(value):
    # Prevent overflow in exp() for very large values.
    value = max(-10.0, min(10.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def random_vector(size):
    return [random.uniform(-0.5, 0.5) for _ in range(size)]


def build_vocabulary(sentences):
    words = sorted({word for sentence in sentences for word in sentence})
    word_to_id = {word: index for index, word in enumerate(words)}
    return words, word_to_id


# -------------------------- WORD2VEC: SKIP-GRAM -----------------------------

def train_word2vec(sentences, vector_size=20, window=2, epochs=120):
    """
    Train a small Skip-gram model with negative sampling.

    The centre word is trained to predict nearby context words.
    This implementation uses only Python lists and the standard library.
    """
    random.seed(10)
    words, word_to_id = build_vocabulary(sentences)
    vocabulary_size = len(words)

    input_vectors = [random_vector(vector_size)
                     for _ in range(vocabulary_size)]
    output_vectors = [[0.0] * vector_size
                      for _ in range(vocabulary_size)]

    training_pairs = []
    for sentence in sentences:
        for position, centre_word in enumerate(sentence):
            start = max(0, position - window)
            end = min(len(sentence), position + window + 1)

            for context_position in range(start, end):
                if context_position != position:
                    training_pairs.append(
                        (word_to_id[centre_word],
                         word_to_id[sentence[context_position]])
                    )

    if not training_pairs:
        return words, word_to_id, {}

    learning_rate = 0.025
    negative_count = 2

    for epoch in range(epochs):
        random.shuffle(training_pairs)

        for centre_id, context_id in training_pairs:
            samples = [(context_id, 1)]

            for _ in range(negative_count):
                negative_id = random.randrange(vocabulary_size)
                if negative_id != context_id:
                    samples.append((negative_id, 0))

            for target_id, label in samples:
                centre_vector = input_vectors[centre_id]
                target_vector = output_vectors[target_id]

                score = dot(centre_vector, target_vector)
                prediction = sigmoid(score)
                error = label - prediction

                old_target = target_vector[:]

                for i in range(vector_size):
                    input_vectors[centre_id][i] += (
                        learning_rate * error * old_target[i]
                    )
                    output_vectors[target_id][i] += (
                        learning_rate * error * centre_vector[i]
                    )

        learning_rate *= 0.98

    embeddings = {
        word: input_vectors[index]
        for word, index in word_to_id.items()
    }
    return words, word_to_id, embeddings


# ------------------------------ GLOVE --------------------------------------

def build_cooccurrence(sentences, word_to_id, window=2):
    """Count how often words occur near one another."""
    counts = defaultdict(float)

    for sentence in sentences:
        for position, word in enumerate(sentence):
            word_id = word_to_id[word]
            start = max(0, position - window)
            end = min(len(sentence), position + window + 1)

            for context_position in range(start, end):
                if context_position == position:
                    continue

                context_id = word_to_id[sentence[context_position]]
                distance = abs(position - context_position)
                counts[(word_id, context_id)] += 1.0 / distance

    return counts


def train_glove(sentences, vector_size=20, window=2, epochs=100):
    """
    Train a simplified GloVe-style model.

    GloVe learns vectors by fitting the logarithm of word co-occurrence counts.
    """
    random.seed(20)
    words, word_to_id = build_vocabulary(sentences)
    size = len(words)

    cooccurrence = build_cooccurrence(sentences, word_to_id, window)

    word_vectors = [random_vector(vector_size) for _ in range(size)]
    context_vectors = [random_vector(vector_size) for _ in range(size)]
    word_bias = [0.0] * size
    context_bias = [0.0] * size

    learning_rate = 0.02

    for epoch in range(epochs):
        for (word_id, context_id), count in cooccurrence.items():
            target = math.log(count)
            prediction = (
                dot(word_vectors[word_id], context_vectors[context_id])
                + word_bias[word_id]
                + context_bias[context_id]
            )
            error = prediction - target

            # GloVe weighting function.
            weight = min(1.0, (count / 5.0) ** 0.75)
            gradient = 2.0 * weight * error

            old_word = word_vectors[word_id][:]
            old_context = context_vectors[context_id][:]

            for i in range(vector_size):
                word_vectors[word_id][i] -= (
                    learning_rate * gradient * old_context[i]
                )
                context_vectors[context_id][i] -= (
                    learning_rate * gradient * old_word[i]
                )

            word_bias[word_id] -= learning_rate * gradient
            context_bias[context_id] -= learning_rate * gradient

    embeddings = {
        word: word_vectors[index]
        for word, index in word_to_id.items()
    }
    return words, word_to_id, embeddings


# -------------------------- EMBEDDING OPERATIONS ---------------------------

def most_similar(word, embeddings, top_n=5):
    """Find words with the highest cosine similarity to the given word."""
    if word not in embeddings:
        return []

    results = []
    for other_word, vector in embeddings.items():
        if other_word != word:
            score = cosine_similarity(embeddings[word], vector)
            results.append((other_word, score))

    return sorted(results, key=lambda item: item[1], reverse=True)[:top_n]


def word_analogy(word_a, word_b, word_c, embeddings):
    """Solve a simple analogy: A - B + C."""
    if not all(word in embeddings for word in (word_a, word_b, word_c)):
        return "One or more analogy words are not in the vocabulary."

    target = [
        embeddings[word_a][i] - embeddings[word_b][i] + embeddings[word_c][i]
        for i in range(len(embeddings[word_a]))
    ]

    results = []
    excluded = {word_a, word_b, word_c}

    for word, vector in embeddings.items():
        if word not in excluded:
            results.append((word, cosine_similarity(target, vector)))

    if not results:
        return "Not enough words to calculate an analogy."

    return max(results, key=lambda item: item[1])


def recommend_products(product_name, embeddings, top_n=3):
    """Recommend products using the average of their description vectors."""
    product_vectors = {}

    for product in PRODUCTS:
        words = tokenize(product["description"])
        known_vectors = [embeddings[word] for word in words
                         if word in embeddings]

        if known_vectors:
            size = len(known_vectors[0])
            average = [
                sum(vector[i] for vector in known_vectors) / len(known_vectors)
                for i in range(size)
            ]
            product_vectors[product["name"]] = average

    if product_name not in product_vectors:
        return []

    results = []
    for name, vector in product_vectors.items():
        if name != product_name:
            score = cosine_similarity(product_vectors[product_name], vector)
            results.append((name, score))

    return sorted(results, key=lambda item: item[1], reverse=True)[:top_n]


# ------------------------------- MAIN --------------------------------------

def show_similar_words(title, embeddings):
    print("\n" + title)
    print("-" * len(title))

    for word in ("phone", "laptop", "music", "health"):
        if word in embeddings:
            print("\nWords similar to:", word)
            for similar_word, score in most_similar(word, embeddings, 3):
                print("  {:15s} {:.3f}".format(similar_word, score))


def main():
    print("=" * 60)
    print("ASSIGNMENT 4: WORD2VEC AND GLOVE")
    print("=" * 60)

    sentences = prepare_corpus()
    print("\nNumber of sample sentences:", len(sentences))

    print("\nTraining Word2Vec Skip-gram model...")
    _, _, word2vec_embeddings = train_word2vec(sentences)
    print("Word2Vec training completed.")

    print("\nTraining GloVe-style model...")
    _, _, glove_embeddings = train_glove(sentences)
    print("GloVe training completed.")

    show_similar_words("WORD2VEC RESULTS", word2vec_embeddings)
    show_similar_words("GLOVE RESULTS", glove_embeddings)

    print("\nWORD ANALOGY")
    print("Analogy: laptop - computer + phone")
    result = word_analogy(
        "laptop", "computer", "phone", word2vec_embeddings
    )
    print("Result:", result)

    print("\nPRODUCT RECOMMENDATIONS")
    print("Products similar to smartphone:")
    for name, score in recommend_products(
        "smartphone", word2vec_embeddings
    ):
        print("  {:20s} {:.3f}".format(name, score))

    print("\nProgram completed successfully.")


if __name__ == "__main__":
    main()
































"""
ASSIGNMENT 4: WORD EMBEDDINGS - WORD2VEC AND GLOVE
-------------------------------------------------
Simple, beginner-friendly implementation using ONLY Python's standard library.

No Gensim, NumPy, SciPy, scikit-learn, Matplotlib, model downloads, or internet
connection are required.

What this program demonstrates:
1. Tokenization of product descriptions.
2. A small Skip-gram Word2Vec-style model trained with negative sampling.
3. A small GloVe-style model trained from word co-occurrence counts.
4. Word similarity, a simple word analogy, and product recommendations.

NOTE:
This is an educational implementation. The small sample corpus is for learning
and demonstration; it is not a replacement for large production embeddings.
"""

# ---------------------------------------------------------------------------
# INSTALLATION COMMANDS
# ---------------------------------------------------------------------------
# No external packages are required.
#
# Windows:
#   1. Install Python 3.8 or later from https://www.python.org/
#   2. Open PowerShell in the folder containing this file.
#   3. Run:
#        python Assignment4_Word2Vec_GloVe_Simple.py
#
# Ubuntu / Linux:
#   1. Install Python:
#        sudo apt update
#        sudo apt install python3
#   2. Run:
#        python3 Assignment4_Word2Vec_GloVe_Simple.py
#
# If the "python" command is not recognized on Windows, try "py".
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# VIVA / ORAL QUESTIONS AND ANSWERS
# ---------------------------------------------------------------------------
#
# Q1. What is a word embedding?
# Ans: A word embedding is a numerical vector representation of a word.
#
# Q2. Why are word embeddings used?
# Ans: They allow computers to process words and capture relationships
#      between words.
#
# Q3. What is Word2Vec?
# Ans: Word2Vec is a method that learns word vectors from surrounding words.
#
# Q4. What are the two Word2Vec architectures?
# Ans: Continuous Bag of Words (CBOW) and Skip-gram.
#
# Q5. What is CBOW?
# Ans: CBOW predicts a target word using its surrounding context words.
#
# Q6. What is Skip-gram?
# Ans: Skip-gram predicts surrounding context words using a target word.
#
# Q7. What is a context window?
# Ans: It is the number of nearby words considered around a target word.
#
# Q8. What is negative sampling?
# Ans: It trains a model using real word pairs and a small number of
#      randomly selected incorrect pairs.
#
# Q9. What is GloVe?
# Ans: GloVe means Global Vectors for Word Representation. It learns vectors
#      using global word co-occurrence statistics.
#
# Q10. What is a co-occurrence matrix?
# Ans: It stores how frequently words appear near other words.
#
# Q11. What is the difference between Word2Vec and GloVe?
# Ans: Word2Vec learns from local context prediction; GloVe learns from
#      aggregated co-occurrence counts.
#
# Q12. What is cosine similarity?
# Ans: It measures the angle-based similarity between two vectors.
#
# Q13. What does a cosine similarity near 1 indicate?
# Ans: The two vectors point in similar directions.
#
# Q14. What is a word analogy?
# Ans: It uses vector arithmetic to represent a relationship between words.
#
# Q15. What is the purpose of tokenization?
# Ans: It splits text into smaller units, such as words.
#
# Q16. What is a vocabulary?
# Ans: It is the set of unique words found in the corpus.
#
# Q17. What is an embedding dimension?
# Ans: It is the number of numerical values in each word vector.
#
# Q18. What is a learning rate?
# Ans: It controls how much model parameters change during training.
#
# Q19. What is an epoch?
# Ans: One complete pass through the training examples.
#
# Q20. What is the purpose of word similarity?
# Ans: It finds words whose vector representations are close.
#
# Q21. How can embeddings support recommendations?
# Ans: Similarity between product vectors can be used to suggest related
#      products.
#
# Q22. What is a corpus?
# Ans: A corpus is a collection of text used for language processing.
#
# Q23. Why is a large corpus useful?
# Ans: It provides more examples of word usage and usually improves
#      embedding quality.
#
# Q24. What is the limitation of this sample program?
# Ans: It uses a small demonstration corpus, so the learned vectors are
#      not suitable for real-world language understanding.
#
# Q25. Does this program need Gensim?
# Ans: No. It uses only Python's standard library.
#
# Q26. Does this program need internet access?
# Ans: No. All sample data is included in the program.
#
# Q27. What is the role of the sigmoid function in Skip-gram?
# Ans: It converts a score into a value between 0 and 1.
#
# Q28. Why are negative examples used?
# Ans: They help the model distinguish actual context words from random words.
#
# Q29. What is the GloVe weighting function used for?
# Ans: It reduces the influence of very rare or excessively frequent
#      co-occurrence counts.
#
# Q30. Can embeddings be used in search engines?
# Ans: Yes. They can help retrieve text or products based on semantic
#      similarity.
# ---------------------------------------------------------------------------
