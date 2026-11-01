import json
import re
from pathlib import Path


# Sample records are used when the JSON dataset is not present.
SAMPLE_RECORDS = [
    {
        "id": "P001",
        "text": (
            "Patient was diagnosed with Type 2 Diabetes and hypertension. "
            "Prescribed Metformin 500 mg and Aspirin 75 mg. "
            "Patient reported dizziness and headache."
        )
    },
    {
        "id": "P002",
        "text": (
            "Patient admitted with fever and cough. Diagnosis was asthma. "
            "Prescribed Amoxicillin 250 mg and Paracetamol 500 mg."
        )
    },
    {
        "id": "P003",
        "text": (
            "Patient with nausea and pain was treated with Ibuprofen 200 mg. "
            "Medical history includes Diabetes Mellitus."
        )
    }
]


# Clinical entity patterns from the assignment.
PATTERNS = {
    "DRUG": (
        r"\b(?:Amoxicillin|Lisinopril|Metformin|Paracetamol|"
        r"Ibuprofen|Aspirin)\b"
    ),
    "DOSAGE": r"\b\d+\s*(?:mg|g|ml)\b",
    "DIAGNOSIS": (
        r"\b(?:hypertension|Diabetes Mellitus|Type 2 Diabetes|"
        r"fever|asthma)\b"
    ),
    "SYMPTOM": r"\b(?:dizziness|nausea|headache|pain|cough)\b"
}


def extract_clinical_entities(text):
    """Extract clinical entities using regular expressions."""
    extracted = {}

    for entity_type, pattern in PATTERNS.items():
        # dict.fromkeys removes duplicates and preserves order.
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        extracted[entity_type] = list(dict.fromkeys(matches))

    return extracted


def create_spacy_pipeline():
    """
    Create a spaCy pipeline with pretrained NER and a custom EntityRuler.
    Returns None if spaCy or its English model is unavailable.
    """
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")

        # Add custom clinical patterns before the statistical NER component.
        ruler = nlp.add_pipe(
            "entity_ruler",
            before="ner",
            config={"overwrite_ents": True}
        )

        patterns = [
            {"label": "DRUG", "pattern": "Amoxicillin"},
            {"label": "DRUG", "pattern": "Lisinopril"},
            {"label": "DRUG", "pattern": "Metformin"},
            {"label": "DRUG", "pattern": "Paracetamol"},
            {"label": "DRUG", "pattern": "Ibuprofen"},
            {"label": "DRUG", "pattern": "Aspirin"},

            {"label": "DIAGNOSIS", "pattern": "hypertension"},
            {"label": "DIAGNOSIS", "pattern": "Diabetes Mellitus"},
            {"label": "DIAGNOSIS", "pattern": "Type 2 Diabetes"},
            {"label": "DIAGNOSIS", "pattern": "fever"},
            {"label": "DIAGNOSIS", "pattern": "asthma"},

            {"label": "SYMPTOM", "pattern": "dizziness"},
            {"label": "SYMPTOM", "pattern": "nausea"},
            {"label": "SYMPTOM", "pattern": "headache"},
            {"label": "SYMPTOM", "pattern": "pain"},
            {"label": "SYMPTOM", "pattern": "cough"}
        ]

        # Add dosage patterns such as 500 mg and 10 ml.
        dosage_pattern = [
            {"LIKE_NUM": True},
            {"LOWER": {"IN": ["mg", "g", "ml"]}}
        ]
        patterns.append({"label": "DOSAGE", "pattern": dosage_pattern})

        ruler.add_patterns(patterns)
        return nlp

    except (ImportError, OSError, Exception) as error:
        print("\nspaCy pipeline is unavailable.")
        print("Reason:", error)
        print("Continuing with regular-expression extraction only.\n")
        return None


def visualize_entities(nlp, text, record_id):
    """Save a displaCy HTML visualization when spaCy is available."""
    if nlp is None:
        return

    from spacy import displacy

    doc = nlp(text)
    html = displacy.render(doc, style="ent", page=True)

    output_file = Path(__file__).resolve().parent / (
        "NER_visualization_" + str(record_id) + ".html"
    )
    output_file.write_text(html, encoding="utf-8")
    print("displaCy visualization saved:", output_file.name)


def load_records():
    """Load the JSON dataset, or use sample records if it is missing."""
    data_file = Path(__file__).resolve().parent / (
        "clinical_discharge_summaries.json"
    )

    if data_file.exists():
        with data_file.open("r", encoding="utf-8") as file:
            records = json.load(file)

        if not isinstance(records, list):
            raise ValueError("JSON data must contain a list of records.")

        return records

    print("Dataset not found. Using built-in sample discharge summaries.")
    return SAMPLE_RECORDS


def main():
    print("=" * 60)
    print("ASSIGNMENT 6: NAMED ENTITY RECOGNITION (NER)")
    print("=" * 60)

    records = load_records()
    nlp = create_spacy_pipeline()

    print("\nNumber of patient records:", len(records))

    for record in records:
        record_id = record.get("id", "Unknown")
        text = record.get("text", "")

        if not text:
            print("\nSkipping record", record_id, "- text is empty.")
            continue

        print("\n" + "-" * 60)
        print("Record ID:", record_id)
        print("Text:", text)

        entities = extract_clinical_entities(text)

        print("\nClinical entities extracted using rules:")
        for entity_type, values in entities.items():
            print(f"{entity_type:10}: {values}")

        if nlp is not None:
            doc = nlp(text)
            print("\nEntities detected by spaCy + EntityRuler:")
            for entity in doc.ents:
                print(f"{entity.text:25} -> {entity.label_}")

            visualize_entities(nlp, text, record_id)

    print("\n" + "=" * 60)
    print("NER PROCESSING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()































"""
ASSIGNMENT 6: NAMED ENTITY RECOGNITION (NER)

Topics covered:
1. Rule-based clinical entity extraction using regular expressions.
2. spaCy pretrained Named Entity Recognition (NER).
3. Custom spaCy EntityRuler for clinical entities.
4. displaCy entity visualization (HTML output).
5. Extract DRUG, DOSAGE, DIAGNOSIS and SYMPTOM from EHR summaries.

------------------------------------------------------------
INSTALLATION COMMANDS
------------------------------------------------------------

WINDOWS (Command Prompt / VS Code Terminal):
    python --version
    python -m pip install --upgrade pip
    python -m pip install spacy
    python -m spacy download en_core_web_sm
    python Assignment6_NER_Simple.py

UBUNTU / LINUX:
    python3 --version
    sudo apt update
    sudo apt install python3 python3-pip
    python3 -m pip install --user spacy
    python3 -m spacy download en_core_web_sm
    python3 Assignment6_NER_Simple.py

NOTE:
- The regular-expression extraction works with Python's standard library.
- spaCy and its English model are optional for the rule-based part.
- If spaCy is not installed, the script still performs clinical extraction.
- Put clinical_discharge_summaries.json in the same folder as this file
  to process your own records. The JSON must contain a list of objects
  with "id" and "text" fields.
"""

# ============================================================
# VIVA QUESTIONS AND ANSWERS
# ============================================================

"""
1. What is Named Entity Recognition (NER)?
   NER is an NLP task that identifies and classifies named entities
   such as people, places, organizations, drugs and medical conditions.

2. What is an entity?
   An entity is a meaningful word or phrase identified in text.

3. What is NLP?
   Natural Language Processing enables computers to process and
   understand human language.

4. What are the entity types in this assignment?
   DRUG, DOSAGE, DIAGNOSIS and SYMPTOM.

5. What is spaCy?
   spaCy is a Python library used for industrial-strength NLP tasks.

6. What is a spaCy pipeline?
   It is a sequence of processing components such as tokenization,
   tagging, parsing and named entity recognition.

7. What is a pretrained NER model?
   It is a model trained on previously labelled text to identify
   entity categories in new text.

8. What is EntityRuler?
   EntityRuler is a spaCy component that identifies entities using
   user-defined patterns.

9. Why is EntityRuler used in this assignment?
   It identifies domain-specific clinical entities such as drug names
   and diagnoses using custom patterns.

10. What is displaCy?
    displaCy is spaCy's built-in visualization tool for displaying
    dependency parses and named entities.

11. What does displaCy style="ent" display?
    It highlights entities and their labels in the input text.

12. What is an EHR?
    EHR stands for Electronic Health Record. It stores a patient's
    clinical and medical information digitally.

13. What is rule-based NER?
    It identifies entities using predefined rules or patterns.

14. Which Python module is used for regular expressions?
    The re module.

15. What does re.findall() do?
    It returns all non-overlapping matches of a pattern in a string.

16. What is a regular expression?
    A regular expression is a pattern used to search and match text.

17. What does \\b mean in a regular expression?
    It represents a word boundary.

18. What does re.IGNORECASE do?
    It makes pattern matching case-insensitive.

19. What is the purpose of the dosage pattern?
    It identifies numeric values followed by units such as mg, g or ml.

20. What is the difference between DRUG and DOSAGE?
    DRUG identifies a medicine name; DOSAGE identifies its quantity
    or measurement, for example Metformin and 500 mg.

21. What is the difference between DIAGNOSIS and SYMPTOM?
    A diagnosis is an identified medical condition; a symptom is
    something experienced or reported by a patient.

22. What is a token in NLP?
    A token is a unit of text, commonly a word or punctuation mark.

23. What is a Doc object in spaCy?
    It is the processed document object containing tokens, entities
    and other linguistic annotations.

24. What is ent.text?
    It returns the text span of a detected entity.

25. What is ent.label_?
    It returns the entity's human-readable label.

26. Can pretrained spaCy NER identify every medical entity?
    No. General pretrained models may not recognize all medical terms.
    Custom rules or domain-trained models can improve coverage.

27. What is the advantage of combining EntityRuler with pretrained NER?
    The pretrained model recognizes general entities, while custom
    rules add domain-specific entity patterns.

28. What is a limitation of rule-based NER?
    It may miss unknown terms and context-dependent meanings unless
    suitable patterns are added.

29. What is JSON?
    JSON (JavaScript Object Notation) is a lightweight data format
    used to store and exchange structured data.

30. What is the output of this assignment?
    It displays extracted clinical entities for each patient record
    and, when spaCy is installed, saves displaCy HTML visualizations.

IMPORTANT:
The patterns cover only the terms listed in this assignment. This is
an educational demonstration, not a clinical diagnosis or a medical
decision-making system.
"""
