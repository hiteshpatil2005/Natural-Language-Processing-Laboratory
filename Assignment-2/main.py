import re
import pandas as pd

# --------------------------------------------------
# Read Dataset
# --------------------------------------------------

df = pd.read_csv('resumes.csv')

print("Original Dataset\n")
print(df)

# --------------------------------------------------
# TextCleaner Class
# --------------------------------------------------

class TextCleaner:

    # Extract Email
    def extract_email(self, text):

        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)

        return ", ".join(emails)

    # Extract URL
    def extract_url(self, text):

        urls = re.findall(r'https?://\S+', text)

        return ", ".join(urls)

    # Extract Phone Number
    def extract_phone(self, text):

        phones = re.findall(r'(\+?\d[\d\-\(\)\s]{8,}\d)', text)

        cleaned = []

        for phone in phones:

            number = re.sub(r'\D', '', phone)

            cleaned.append(number)

        return ", ".join(cleaned)

    # Extract Hashtags
    def extract_hashtag(self, text):

        hashtags = re.findall(r'#(\w+)', text)

        return ", ".join(hashtags)

    # Remove HTML Tags
    def remove_html(self, text):

        return re.sub(r'<.*?>', '', text)

    # Remove JSON Tags
    def remove_json(self, text):

        return re.sub(r'\{.*?\}', '', text)

    # Extract Experience
    def extract_experience(self, text):

        match = re.search(r'(\d+)\s+years', text, re.I)

        if match:

            return match.group(1)

        return ""

    # Extract Name
    def extract_name(self, text):

        words = text.split()

        return " ".join(words[:2])

    # Extract Skills
    def extract_skills(self, text):

        skills = ['Python','Java','SQL','Machine Learning',
                  'Data Science','AI','NLP',
                  'Deep Learning','React','HTML','JavaScript']

        found = []

        for skill in skills:

            if skill.lower() in text.lower():

                found.append(skill)

        return ", ".join(found)

# --------------------------------------------------

cleaner = TextCleaner()

# --------------------------------------------------
# Apply Functions
# --------------------------------------------------

df["Name"] = df["resume"].apply(cleaner.extract_name)

df["Email"] = df["resume"].apply(cleaner.extract_email)

df["Phone"] = df["resume"].apply(cleaner.extract_phone)

df["Experience"] = df["resume"].apply(cleaner.extract_experience)

df["Skills"] = df["resume"].apply(cleaner.extract_skills)

df["URL"] = df["resume"].apply(cleaner.extract_url)

df["Hashtags"] = df["resume"].apply(cleaner.extract_hashtag)

# Remove HTML

df["Clean_Text"] = df["resume"].apply(cleaner.remove_html)

# Remove JSON

df["Clean_Text"] = df["Clean_Text"].apply(cleaner.remove_json)

# --------------------------------------------------
# Display Results
# --------------------------------------------------

print("\n\nExtracted Information\n")

print(df[["Name",
          "Email",
          "Phone",
          "Experience",
          "Skills",
          "URL",
          "Hashtags"]])

# --------------------------------------------------
# Save CSV
# --------------------------------------------------

df.to_csv("extracted_resume_data.csv", index=False)

print("\nData Saved Successfully!")