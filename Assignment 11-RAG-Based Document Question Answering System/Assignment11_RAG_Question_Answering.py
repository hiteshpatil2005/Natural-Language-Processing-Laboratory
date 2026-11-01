import json
import os
import re
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


HANDBOOK_FILE = "university_handbook.txt"
OPTIONAL_TEST_FILE = "rag_test_questions.json"
TOP_K = 3
MIN_SIMILARITY = 0.05


class RAGQuestionAnsweringSystem:
    """A simple Retrieval-Augmented Generation-style document QA pipeline."""

    def __init__(self, handbook_filepath: str):
        if not os.path.isfile(handbook_filepath):
            raise FileNotFoundError(
                f"Handbook file not found: {handbook_filepath}"
            )

        with open(handbook_filepath, "r", encoding="utf-8") as file:
            self.raw_text = file.read()

        if not self.raw_text.strip():
            raise ValueError("The handbook file is empty.")

        self.chunks = self.chunk_document(self.raw_text)
        if not self.chunks:
            raise ValueError("No usable text chunks were found in the handbook.")

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000
        )
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        except ValueError as error:
            raise ValueError(
                "The handbook does not contain enough searchable text. "
                "Check that university_handbook.txt contains readable words."
            ) from error

    @staticmethod
    def chunk_document(raw_text: str) -> List[str]:
        """
        Split by SECTION headings when present. If the file has no SECTION
        markers, split by blank lines and combine short paragraphs.
        """
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()

        # Split at headings such as SECTION 1, Section 2, etc.
        section_parts = re.split(
            r"(?im)(?=^\s*SECTION\s+\w+)",
            text
        )
        section_chunks = [part.strip() for part in section_parts if part.strip()]

        if len(section_chunks) > 1:
            return section_chunks

        # General text fallback: paragraph-based chunking.
        paragraphs = [
            p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()
        ]

        if not paragraphs:
            return [text]

        chunks = []
        current = []
        current_length = 0
        target_chars = 900

        for paragraph in paragraphs:
            if current and current_length + len(paragraph) > target_chars:
                chunks.append("\n\n".join(current))
                current = []
                current_length = 0
            current.append(paragraph)
            current_length += len(paragraph)

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def retrieve_context(self, query: str, top_k: int = TOP_K) -> List[dict]:
        """Retrieve the top-k handbook chunks by cosine similarity."""
        if not query or not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(
            query_vector, self.tfidf_matrix
        )[0]

        top_indices = similarities.argsort()[::-1][:max(1, top_k)]
        results = []

        for index in top_indices:
            score = float(similarities[index])
            results.append({
                "chunk_id": int(index + 1),
                "similarity": score,
                "text": self.chunks[index]
            })

        return results

    def answer_query(self, query: str, top_k: int = TOP_K) -> dict:
        """
        Return a context-grounded extractive answer.
        If similarity is too low, avoid guessing and report no relevant context.
        """
        retrieved = self.retrieve_context(query, top_k)

        if not retrieved:
            return {
                "query": query,
                "answer": "Please enter a valid question.",
                "confidence_score": 0.0,
                "retrieved_contexts": []
            }

        best = retrieved[0]
        if best["similarity"] < MIN_SIMILARITY:
            answer = (
                "I could not find sufficiently relevant information in the "
                "provided handbook. Please check the document or rephrase "
                "your question."
            )
        else:
            answer = self.extract_relevant_sentences(query, best["text"])

        return {
            "query": query,
            "answer": answer,
            "confidence_score": round(best["similarity"], 4),
            "retrieved_contexts": retrieved
        }

    @staticmethod
    def extract_relevant_sentences(query: str, context: str) -> str:
        """Select sentence(s) with the greatest word overlap with the query."""
        sentences = re.split(r"(?<=[.!?])\s+|\n+", context)
        sentences = [s.strip() for s in sentences if s.strip()]

        query_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", query.lower()))
        if not sentences:
            return context.strip()

        scored = []
        for sentence in sentences:
            sentence_words = set(
                re.findall(r"\b[a-zA-Z0-9]+\b", sentence.lower())
            )
            overlap = len(query_words.intersection(sentence_words))
            scored.append((overlap, sentence))

        best_score = max(score for score, _ in scored)
        if best_score == 0:
            # Return the retrieved context rather than inventing an answer.
            return context.strip()

        selected = [
            sentence for score, sentence in scored if score == best_score
        ][:3]
        return " ".join(selected)


def load_optional_test_questions(filepath: str) -> List[dict]:
    """Load optional question/reference-answer pairs for ROUGE and BLEU."""
    if not os.path.isfile(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("The optional test JSON must contain a list.")

    valid = []
    for row in records:
        if (
            isinstance(row, dict)
            and isinstance(row.get("question"), str)
            and isinstance(row.get("reference_answer"), str)
            and row["question"].strip()
            and row["reference_answer"].strip()
        ):
            valid.append(row)

    return valid


def calculate_rouge_l(reference: str, candidate: str) -> float:
    """Calculate ROUGE-L F1 using longest common subsequence."""
    ref = reference.lower().split()
    cand = candidate.lower().split()

    if not ref or not cand:
        return 0.0

    previous = [0] * (len(cand) + 1)
    for ref_word in ref:
        current = [0]
        for j, cand_word in enumerate(cand, start=1):
            if ref_word == cand_word:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current

    lcs = previous[-1]
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_bleu(reference: str, candidate: str) -> float:
    """Calculate sentence BLEU with NLTK when installed."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        return -1.0

    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()

    if not reference_tokens or not candidate_tokens:
        return 0.0

    return float(sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        smoothing_function=SmoothingFunction().method1
    ))


def evaluate_with_references(rag_system: RAGQuestionAnsweringSystem,
                             test_records: List[dict]) -> None:
    """Evaluate answers against user-supplied reference answers."""
    if not test_records:
        print("\nROUGE/BLEU evaluation skipped.")
        print(
            f"To enable it, provide {OPTIONAL_TEST_FILE} with question and "
            "reference_answer fields."
        )
        return

    rouge_scores = []
    bleu_scores = []

    print("\n--- Reference-based Evaluation ---")
    for index, record in enumerate(test_records, start=1):
        result = rag_system.answer_query(record["question"])
        candidate = result["answer"]
        reference = record["reference_answer"]

        rouge = calculate_rouge_l(reference, candidate)
        bleu = calculate_bleu(reference, candidate)
        rouge_scores.append(rouge)
        if bleu >= 0:
            bleu_scores.append(bleu)

        print(f"\nTest {index}: {record['question']}")
        print(f"Generated/extracted answer: {candidate}")
        print(f"Reference answer: {reference}")
        print(f"ROUGE-L F1: {rouge:.4f}")
        if bleu >= 0:
            print(f"BLEU: {bleu:.4f}")
        else:
            print("BLEU: Install nltk to calculate this metric.")

    print("\nAverage ROUGE-L F1:", round(sum(rouge_scores) / len(rouge_scores), 4))
    if bleu_scores:
        print("Average BLEU:", round(sum(bleu_scores) / len(bleu_scores), 4))


def main():
    print("=" * 68)
    print("ASSIGNMENT 11: RAG-BASED DOCUMENT QUESTION ANSWERING SYSTEM")
    print("=" * 68)

    folder = os.path.dirname(os.path.abspath(__file__))
    handbook_path = os.path.join(folder, HANDBOOK_FILE)

    try:
        rag_system = RAGQuestionAnsweringSystem(handbook_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"\nInput error: {error}")
        return

    print(f"\nLoaded handbook: {HANDBOOK_FILE}")
    print(f"Indexed {len(rag_system.chunks)} document chunks.")
    print("Vector store: TF-IDF")
    print("Similarity measure: Cosine similarity")
    print(f"Retrieval setting: Top-{TOP_K}")

    queries = [
        "When do end-semester examinations commence?",
        "What are the postgraduate admission eligibility criteria?",
        "What is the curfew time for hostel residents?"
    ]

    print("\n--- University Student-Support Question Answering ---")
    for query in queries:
        result = rag_system.answer_query(query)
        print("\nUser Query:", result["query"])
        print("Similarity score:", result["confidence_score"])
        print("Answer:", result["answer"])
        print("Retrieved context:")
        for item in result["retrieved_contexts"]:
            print(
                f"  Chunk {item['chunk_id']} "
                f"(score={item['similarity']:.4f})"
            )
            print(" ", item["text"][:350].replace("\n", " "), "...")

    # Interactive question loop
    print("\n--- Ask Your Own Handbook Questions ---")
    print("Type 'exit' to finish.")
    while True:
        question = input("\nEnter your question: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            print("Please enter a question.")
            continue

        result = rag_system.answer_query(question)
        print("\nAnswer:", result["answer"])
        print("Similarity score:", result["confidence_score"])
        print("\nTop retrieved passages:")
        for item in result["retrieved_contexts"]:
            print(
                f"\n[Chunk {item['chunk_id']} | "
                f"Similarity: {item['similarity']:.4f}]"
            )
            print(item["text"][:700])

    test_path = os.path.join(folder, OPTIONAL_TEST_FILE)
    try:
        test_records = load_optional_test_questions(test_path)
        evaluate_with_references(rag_system, test_records)
    except (ValueError, json.JSONDecodeError) as error:
        print(f"\nEvaluation input error: {error}")

    print("\nAssignment 11 completed.")


if __name__ == "__main__":
    main()


















































"""
ASSIGNMENT 11: RAG-BASED DOCUMENT QUESTION ANSWERING SYSTEM

Topics:
- Chunking and indexing institutional handbook documents
- TF-IDF vector store and cosine-similarity retrieval
- Top-k context retrieval for student questions
- Context-grounded answer generation
- Optional ROUGE and BLEU evaluation using reference answers

EXTERNAL DATA FILES
-------------------
Keep your existing university_handbook.txt in the same folder as this script.
This program does NOT create or insert sample handbook content.

Optional evaluation file: rag_test_questions.json
Use this format if you have reference answers for evaluation:
[
    {
        "question": "When do examinations start?",
        "reference_answer": "Examinations begin in the second week of May."
    }
]

INSTALLATION
------------
Windows:
    py -3.13 -m venv nlp_env
    nlp_env\\Scripts\\activate
    python -m pip install --upgrade pip
    pip install scikit-learn

Ubuntu / Linux:
    python3 -m venv nlp_env
    source nlp_env/bin/activate
    python -m pip install --upgrade pip
    pip install scikit-learn

Optional ROUGE/BLEU package:
    pip install nltk

RUN
---
    python Assignment11_RAG_Question_Answering.py

IMPORTANT
---------
- The handbook file is required and must be named university_handbook.txt.
- This implementation uses TF-IDF and cosine similarity as a lightweight
  vector retrieval method; it does not require FAISS or an embedding API.
- The answer is extracted from the retrieved handbook context. It does not
  call an external generative LLM. This keeps answers grounded in your file.
- ROUGE/BLEU are only calculated when rag_test_questions.json contains
  reference answers. Without references, the program reports that evaluation
  cannot be calculated; it does not invent scores.
"""



# --------------------------- 30 VIVA QUESTIONS WITH ANSWERS ------------
"""
1. What is RAG?
   Retrieval-Augmented Generation combines document retrieval with answer
   generation grounded in the retrieved information.

2. Why is RAG used?
   It lets a system answer questions using a supplied knowledge source.

3. What are the main stages of RAG?
   Document loading, chunking, indexing, retrieval and answer generation.

4. What is document chunking?
   Dividing a large document into smaller text sections.

5. Why is chunking important?
   Smaller sections make relevant information easier to retrieve.

6. What is a vector store?
   A structure used to store document representations for similarity search.

7. What is TF-IDF?
   Term Frequency-Inverse Document Frequency represents text using weighted
   word importance.

8. What is term frequency?
   It measures how often a term appears in a document.

9. What is inverse document frequency?
   It reduces the weight of terms that appear in many documents.

10. What is cosine similarity?
    A measure of similarity based on the angle between two vectors.

11. What does top-k retrieval mean?
    Returning the k highest-scoring document chunks for a query.

12. What is a query?
    The question or search text submitted by the user.

13. What is a retriever?
    A component that finds relevant documents or chunks for a query.

14. What is context grounding?
    Producing an answer using evidence from retrieved source material.

15. What is hallucination?
    A model producing unsupported or fabricated information.

16. How can RAG reduce hallucinations?
    It supplies relevant source context and can require answers to stay within it.

17. What is an embedding?
    A numerical representation of text in a vector space.

18. Does this implementation use neural embeddings?
    No. It uses TF-IDF vectors as a lightweight retrieval representation.

19. What is FAISS?
    A library for efficient similarity search over dense vectors.

20. Does this implementation require FAISS?
    No. It uses scikit-learn TF-IDF and cosine similarity.

21. What is a knowledge base?
    A collection of documents used as the source of information for a system.

22. What is ROUGE?
    A family of metrics that compares generated text with reference text using
    overlapping units.

23. What is BLEU?
    A metric that evaluates generated text using n-gram precision against
    reference text.

24. Why are reference answers needed for ROUGE and BLEU?
    These metrics compare a candidate answer with a known reference answer.

25. What is retrieval accuracy?
    It describes how effectively a retriever finds relevant source material.

26. What is semantic search?
    Searching for content based on meaning or relatedness rather than only
    exact keyword matches.

27. What is the difference between retrieval and generation?
    Retrieval finds source information; generation formulates a response from it.

28. Why is a similarity threshold useful?
    It helps avoid answering when retrieved content is not sufficiently relevant.

29. What are limitations of TF-IDF retrieval?
    It relies on word overlap and may miss semantic relationships expressed
    with different vocabulary.

30. Give one application of a RAG system.
    A university chatbot that answers student questions using an official
    handbook.
"""
