import re
import emoji
import nltk
import spacy
from transformers import AutoTokenizer

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag


# 1. Download required NLTK resources
nltk_resources = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng"
]

for resource in nltk_resources:
    nltk.download(resource, quiet=True)


# 2. Load spaCy and multilingual BERT tokenizer
nlp = spacy.load("en_core_web_sm")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


# 3. Sample multilingual and noisy text dataset
dataset = [
    "Breaking News!! India wins the match 🇮🇳🏏 #Cricket https://news.com",
    "OMG 😂😂 This fake news is spreading everywhere!!",
    "आज मौसम बहुत अच्छा है 😊",
    "COVID-19 cases are increasing rapidly in Europe.",
    "@john Check this out!! https://abc.com #Breaking",
    "Yeh news bilkul fake hai!! 😡😡"
]


# 4. Text cleaning
def clean_text(text):
    text = re.sub(r"http\S+", "", text)       # Remove URLs
    text = re.sub(r"@\w+", "", text)          # Remove mentions
    text = text.replace("#", "")              # Keep hashtag words
    text = emoji.replace_emoji(text, replace="")  # Remove emojis
    text = re.sub(r"\s+", " ", text)           # Remove extra spaces
    return text.strip()


clean_corpus = [clean_text(text) for text in dataset]

print("\n========== CLEANED TEXT ==========")
for text in clean_corpus:
    print(text)


# 5. Tokenization
regex_tokenizer = RegexpTokenizer(r"\w+")

print("\n========== WHITESPACE TOKENIZATION ==========")
for text in clean_corpus:
    print(text.split())

print("\n========== RULE-BASED TOKENIZATION ==========")
for text in clean_corpus:
    print(regex_tokenizer.tokenize(text))

print("\n========== SUBWORD TOKENIZATION (MULTILINGUAL BERT) ==========")
for text in clean_corpus:
    print(bert_tokenizer.tokenize(text))


# 6. Stop-word removal
stop_words = set(stopwords.words("english"))

print("\n========== STOP-WORD REMOVAL ==========")
for text in clean_corpus:
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word.lower() not in stop_words]
    print(filtered_words)


# 7. Stemming using Porter and Snowball
porter = PorterStemmer()
snowball = SnowballStemmer("english")

print("\n========== PORTER STEMMING ==========")
for text in clean_corpus:
    words = regex_tokenizer.tokenize(text)
    print([porter.stem(word) for word in words])

print("\n========== SNOWBALL STEMMING ==========")
for text in clean_corpus:
    words = regex_tokenizer.tokenize(text)
    print([snowball.stem(word) for word in words])


# 8. Lemmatization using WordNet
lemmatizer = WordNetLemmatizer()

print("\n========== WORDNET LEMMATIZATION ==========")
for text in clean_corpus:
    words = regex_tokenizer.tokenize(text)
    print([lemmatizer.lemmatize(word) for word in words])


# 9. POS tagging using NLTK
print("\n========== NLTK POS TAGGING ==========")
for text in clean_corpus:
    tokens = word_tokenize(text)
    print(pos_tag(tokens))


# 10. spaCy tokenization, lemmatization and POS tagging
print("\n========== SPACY PROCESSING ==========")
for text in clean_corpus:
    doc = nlp(text)
    print("\nText:", text)
    print("Tokens:", [token.text for token in doc])
    print("Lemmas:", [token.lemma_ for token in doc])
    print("POS tags:", [(token.text, token.pos_) for token in doc])


# 11. Compare NLTK and spaCy on the same original (raw) text
raw_text = dataset[0]

print("\n========== NLTK VS SPACY COMPARISON ==========")
print("Original raw text:", raw_text)

nltk_tokens = word_tokenize(raw_text)
print("\nNLTK tokens:", nltk_tokens)
print("NLTK POS tags:", pos_tag(nltk_tokens))

spacy_doc = nlp(raw_text)
print("\nspaCy tokens:", [token.text for token in spacy_doc])
print("spaCy lemmas:", [token.lemma_ for token in spacy_doc])
print("spaCy POS tags:", [(token.text, token.pos_) for token in spacy_doc])

print("\nText preprocessing pipeline completed successfully.")

"""
ASSIGNMENT 1: TEXT PREPROCESSING PIPELINE
PCIT404 - Natural Language Processing Laboratory

============================================================
INSTALLATION / DEPENDENCY COMMANDS
============================================================

A) WINDOWS (Command Prompt or PowerShell)
-----------------------------------------
1. Create and activate a virtual environment:

   py -3.11 -m venv venv
   venv\\Scripts\\activate

2. Upgrade pip and install Python libraries:

   python -m pip install --upgrade pip
   pip install nltk spacy emoji transformers torch

3. Download the spaCy English model:

   python -m spacy download en_core_web_sm

4. Run the program:

   python Assignment1_Text_Preprocessing.py


B) UBUNTU / LINUX (Terminal)
----------------------------
1. Install Python and virtual-environment support:

   sudo apt update
   sudo apt install python3 python3-venv python3-pip

2. Create and activate a virtual environment:

   python3 -m venv venv
   source venv/bin/activate

3. Upgrade pip and install Python libraries:

   python -m pip install --upgrade pip
   pip install nltk spacy emoji transformers torch

4. Download the spaCy English model:

   python -m spacy download en_core_web_sm

5. Run the program:

   python Assignment1_Text_Preprocessing.py


NOTES:
- Python 3.10 or 3.11 is recommended for easier library compatibility.
- Internet access is required for the first run to download NLTK data
  and the multilingual BERT tokenizer.
- Keep the virtual environment activated whenever you run this program.
- If your Linux system uses the command 'python3' instead of 'python',
  use 'python3 Assignment1_Text_Preprocessing.py' to run the file.
============================================================
"""
# ============================================================
# VIVA / ORAL EXAMINATION THEORY
# Read these notes after running the practical.
# ============================================================

"""
1. What is NLP?
   NLP (Natural Language Processing) is a branch of Artificial
   Intelligence that helps computers process and understand human
   language.

2. What is text preprocessing?
   Text preprocessing means cleaning and preparing raw text before
   using it in an NLP application or machine-learning model.

3. What is a corpus?
   A corpus is a collection of text documents used for language
   processing and analysis.

4. What is tokenization?
   Tokenization is the process of splitting text into smaller units
   called tokens, such as words or subwords.

5. What are the tokenization methods used here?
   - Whitespace tokenization: splits text wherever whitespace occurs.
   - Rule-based tokenization: uses a pattern or rule to identify tokens.
   - Subword tokenization: splits words into smaller model vocabulary
     units. This practical uses the multilingual BERT tokenizer.

6. What are stop words?
   Stop words are common words such as 'is', 'the', 'are' and 'and'
   that are often removed when they do not add useful information
   for a particular NLP task.

7. What is stemming?
   Stemming reduces a word by removing or changing its ending.
   Example: 'playing' may become 'play'. A stem is not always a
   valid dictionary word.

8. Which stemmers are used in this practical?
   PorterStemmer and SnowballStemmer from NLTK.

9. What is lemmatization?
   Lemmatization converts a word to its dictionary base form (lemma).
   Example: 'cars' becomes 'car'. The result can depend on the word's
   grammatical role and the lemmatizer's linguistic information.

10. Difference between stemming and lemmatization?
    Stemming applies simpler word-ending rules and may produce a
    non-dictionary form. Lemmatization uses vocabulary and linguistic
    information to produce a meaningful base form.

11. What is WordNet?
    WordNet is a lexical database of English words and their semantic
    relationships. NLTK's WordNetLemmatizer uses it for lemmatization.

12. What is POS tagging?
    POS means Part of Speech. POS tagging assigns a grammatical label
    to each word, such as noun, verb, adjective or adverb.

13. Which function performs POS tagging in NLTK?
    nltk.pos_tag() assigns part-of-speech tags to a list of tokens.

14. What is spaCy?
    spaCy is a Python NLP library that provides tools such as
    tokenization, lemmatization, POS tagging and named entity recognition.

15. What is NLTK?
    NLTK (Natural Language Toolkit) is a Python library containing
    tools and datasets for learning and performing NLP tasks.

16. Difference between NLTK and spaCy?
    NLTK provides many NLP algorithms and learning resources.
    spaCy provides efficient, pretrained NLP pipelines designed for
    practical applications.

17. What is multilingual BERT?
    Multilingual BERT is a pretrained transformer model that supports
    multiple languages. In this practical, its tokenizer is used for
    subword tokenization; the model itself is not fine-tuned.

18. Why do we clean URLs, mentions and emojis?
    They may be unwanted for a particular text-analysis task. Cleaning
    them can reduce noise. However, emojis and hashtags may carry
    sentiment or meaning, so they should only be removed when suitable.

19. What is a regular expression?
    A regular expression (regex) is a pattern used to search, match,
    remove or replace text. Python's re module provides regex functions.

20. Why compare NLTK and spaCy?
    Their tokenization and linguistic processing may differ. Comparing
    their output helps understand the tools and choose one suitable
    for an NLP task.

21. What does the spaCy model en_core_web_sm mean?
    It is spaCy's small English pipeline containing trained components
    for common NLP tasks, including POS tagging and lemmatization.

22. Does this program perform full Hindi POS tagging?
    No. The selected spaCy model is English-only. The multilingual
    BERT tokenizer can split Hindi text into subword tokens, but that
    does not provide Hindi POS tagging or lemmatization.

23. What are the main libraries used?
    - re: regular-expression-based text cleaning.
    - emoji: removing emoji characters.
    - NLTK: tokenization, stop words, stemming, lemmatization and POS.
    - spaCy: English NLP processing.
    - Transformers: loading the multilingual BERT tokenizer.

24. What is the output of this practical?
    The program displays cleaned text, different tokenization outputs,
    stop-word removal, stemming, lemmatization, POS tags and a
    comparison of NLTK and spaCy.

============================================================
IMPORTANT FUNCTIONS TO REMEMBER
============================================================
re.sub()                         - replace text matching a pattern
text.split()                     - whitespace tokenization
RegexpTokenizer()                - rule-based tokenization
AutoTokenizer.from_pretrained()  - load a pretrained tokenizer
stopwords.words("english")       - get English stop words
PorterStemmer()                  - create Porter stemmer
SnowballStemmer("english")       - create Snowball stemmer
WordNetLemmatizer()              - create WordNet lemmatizer
pos_tag(tokens)                  - perform NLTK POS tagging
spacy.load("en_core_web_sm")     - load spaCy English model
"""
