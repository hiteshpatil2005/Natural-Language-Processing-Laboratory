import re
import pandas as pd


# 1. Sample noisy text (Twitter / Reddit-style)
posts = [
    "Contact me at test@gmail.com #Python https://example.com 😊",
    "Visit <b>our website</b> for details! #NLP",
    "New update {\"status\":\"available\"} call +91 98765-43210",
]

# 2. TextCleaner class
class TextCleaner:

    def extract_emails(self, text):
        return ", ".join(re.findall(r'[\w.-]+@[\w.-]+\.\w+', text))

    def extract_urls(self, text):
        return ", ".join(re.findall(r'https?://\S+', text))

    def extract_phones(self, text):
        phones = re.findall(r'\+?\d[\d\s()-]{8,}\d', text)
        return ", ".join(re.sub(r'\D', '', phone) for phone in phones)

    def extract_hashtags(self, text):
        return ", ".join(re.findall(r'#(\w+)', text))

    def remove_html(self, text):
        return re.sub(r'<[^>]+>', '', text)

    def remove_json(self, text):
        return re.sub(r'\{.*?\}', '', text)

    def clean_text(self, text):
        text = self.remove_html(text)
        text = self.remove_json(text)
        return re.sub(r'\s+', ' ', text).strip()

    def extract_name(self, text):
        # Simple assumption: first two words are the candidate's name.
        words = text.split()
        return " ".join(words[:2])

    def extract_experience(self, text):
        match = re.search(r'(\d+)\s*\+?\s*years?', text, re.IGNORECASE)
        return match.group(1) if match else ""

    def extract_skills(self, text):
        skill_list = [
            "Python", "Java", "SQL", "Machine Learning", "Data Science",
            "AI", "NLP", "Deep Learning", "React", "HTML", "JavaScript"
        ]
        return ", ".join(
            skill for skill in skill_list if re.search(
                r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE
            )
        )


cleaner = TextCleaner()


# 3. Test TextCleaner on noisy social-media text
print("\n========== TEXT CLEANING: NOISY POSTS ==========")

for post in posts:
    print("\nOriginal:", post)
    print("Email:", cleaner.extract_emails(post))
    print("URL:", cleaner.extract_urls(post))
    print("Phone:", cleaner.extract_phones(post))
    print("Hashtags:", cleaner.extract_hashtags(post))
    print("Clean text:", cleaner.clean_text(post))


# 4. Create sample resume dataset if it does not exist
sample_resumes = [
    "Asha Patil | Email: asha@gmail.com | Phone: +91 98765-43210 | 2 years experience | Python, SQL, Machine Learning",
    "Rahul Sharma | Email: rahul.sharma@yahoo.com | Phone: 9876543210 | 1 year experience | Java, HTML, JavaScript, React",
    "Neha Joshi | Email: neha@example.com | Phone: +91-91234-56789 | 3 years experience | Data Science, AI, NLP, Python",
]

try:
    df = pd.read_csv("resumes.csv")
except FileNotFoundError:
    df = pd.DataFrame({"resume": sample_resumes})
    df.to_csv("resumes.csv", index=False)
    print("\nSample resumes.csv created.")


# 5. Extract details from resumes
if "resume" not in df.columns:
    raise ValueError("The input CSV must contain a column named 'resume'.")

df["Name"] = df["resume"].fillna("").apply(cleaner.extract_name)
df["Email"] = df["resume"].fillna("").apply(cleaner.extract_emails)
df["Phone"] = df["resume"].fillna("").apply(cleaner.extract_phones)
df["Experience"] = df["resume"].fillna("").apply(cleaner.extract_experience)
df["Skills"] = df["resume"].fillna("").apply(cleaner.extract_skills)
df["URL"] = df["resume"].fillna("").apply(cleaner.extract_urls)
df["Hashtags"] = df["resume"].fillna("").apply(cleaner.extract_hashtags)
df["Clean_Text"] = df["resume"].fillna("").apply(cleaner.clean_text)


# 6. Display extracted information
print("\n========== EXTRACTED RESUME INFORMATION ==========")
print(df[["Name", "Email", "Phone", "Experience", "Skills", "URL", "Hashtags"]].to_string(index=False))


# 7. Save the structured output
df.to_csv("extracted_resume_data.csv", index=False)
print("\nStructured data saved to extracted_resume_data.csv")
print("Assignment 2 completed successfully.")






"""
ASSIGNMENT 2: REGULAR EXPRESSIONS AND TEXT CLEANING
PCIT404 - Natural Language Processing Laboratory

============================================================
INSTALLATION / DEPENDENCY COMMANDS
============================================================

A) WINDOWS (Command Prompt or PowerShell)
-----------------------------------------
1. Create and activate a virtual environment:

   py -3.11 -m venv venv
   venv\Scripts\activate

2. Install the required library:

   python -m pip install --upgrade pip
   pip install pandas

3. Run the program:

   python Assignment2_Regular_Expressions.py


B) UBUNTU / LINUX (Terminal)
----------------------------
1. Install Python and virtual-environment support:

   sudo apt update
   sudo apt install python3 python3-venv python3-pip

2. Create and activate a virtual environment:

   python3 -m venv venv
   source venv/bin/activate

3. Install the required library:

   python -m pip install --upgrade pip
   pip install pandas

4. Run the program:

   python Assignment2_Regular_Expressions.py


NOTES:
- Python 3.10 or 3.11 is recommended.
- The 're' module is built into Python, so it does not need installation.
- pandas is used to store and export extracted resume details.
- This program creates a small sample resumes.csv file automatically
  if the file is not already present in the current folder.
============================================================
"""

# ============================================================
# VIVA / ORAL EXAMINATION THEORY
# ============================================================

"""
1. What is a regular expression?
   A regular expression (regex) is a pattern used to search, match,
   extract or replace text.

2. Which Python module is used for regular expressions?
   The 're' module.

3. Do we need to install the re module?
   No. It is included with Python.

4. What is re.findall()?
   It finds all non-overlapping matches of a pattern and returns them
   as a list.

5. What is re.search()?
   It searches for the first match of a pattern in a string.

6. What is re.sub()?
   It replaces text matching a pattern with another value.

7. What is the purpose of the TextCleaner class?
   It groups text-cleaning and information-extraction functions in
   one reusable class.

8. What is an email regex used in this practical?
   r'[\\w.-]+@[\\w.-]+\\.\\w+'

9. How are URLs detected?
   Using the pattern r'https?://\\S+' to find HTTP or HTTPS links.

10. How are phone numbers normalized?
    The program finds phone-like strings and removes all non-digit
    characters using re.sub(r'\\D', '', phone).

11. What does \\d mean in regex?
    It matches a digit from 0 to 9.

12. What does \\w mean?
    It matches a word character: letters, digits or underscore.

13. What does \\s mean?
    It matches whitespace such as spaces, tabs and newlines.

14. What does \\D mean?
    It matches any character that is not a digit.

15. What does * mean in regex?
    It means zero or more repetitions of the preceding pattern.

16. What does + mean in regex?
    It means one or more repetitions of the preceding pattern.

17. What does re.IGNORECASE do?
    It makes pattern matching case-insensitive.

18. How are hashtags extracted?
    The pattern r'#(\\w+)' finds the word after the # symbol.

19. How are HTML tags removed?
    re.sub(r'<[^>]+>', '', text) removes text enclosed in angle brackets
    that matches an HTML-tag-like pattern.

20. How is JSON-like text removed in this practical?
    re.sub(r'\\{.*?\\}', '', text) removes text between braces on a
    single line. It is a simple pattern, not a complete JSON parser.

21. What is pandas?
    Pandas is a Python library used for handling tabular data with
    DataFrames and Series.

22. What is a DataFrame?
    A DataFrame is a two-dimensional table with rows and columns.

23. What does pd.read_csv() do?
    It reads data from a CSV file into a pandas DataFrame.

24. What does df.to_csv() do?
    It saves a DataFrame into a CSV file.

25. What is resume parsing?
    Resume parsing is the process of extracting structured details
    such as name, email, phone number, experience and skills from
    unstructured resume text.

26. What is the output file of this practical?
    extracted_resume_data.csv

27. Why is the extracted data stored in CSV format?
    CSV is a simple tabular format that can be opened in spreadsheet
    software and used for further analysis.

28. What are the limitations of this simple resume parser?
    The name extraction assumes the first two words are the name.
    Regex may miss differently formatted phone numbers, emails or
    experience details. Skill extraction only checks the predefined
    skill list. Real resume parsing needs more advanced methods.

29. What is the difference between extraction and cleaning?
    Extraction finds and returns specific information. Cleaning
    removes unwanted text or normalizes the remaining text.

30. What is the main objective of this assignment?
    To use regular expressions and a reusable TextCleaner class to
    clean noisy text and extract structured information from resumes.

============================================================
IMPORTANT FUNCTIONS TO REMEMBER
============================================================
re.findall()             - find all matches
re.search()              - find the first match
re.sub()                 - replace matching text
re.IGNORECASE            - ignore uppercase/lowercase differences
pd.read_csv()             - read a CSV file
pd.DataFrame()            - create a table
df.to_csv()               - save a DataFrame as CSV
str.split()               - split text into words
re.escape()               - escape special regex characters
============================================================
"""
