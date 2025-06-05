import os
import spacy
import pdfplumber
import docx
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util

# Load NLP tools
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# === FILE LOADING ===
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def get_text_from_file(path):
    if path.endswith(".pdf"):
        return extract_text_from_pdf(path)
    elif path.endswith(".docx"):
        return extract_text_from_docx(path)
    else:
        raise ValueError("Unsupported file type. Use PDF or DOCX.")

# === EXTRACTION ===
def extract_skills_keybert(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=15)
    return [kw[0] for kw in keywords]

def extract_entities_spacy(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'SKILL', 'PERSON', 'DATE', 'NORP']]

# === SIMILARITY ===
def compare_similarity(text1, text2):
    emb1 = embedder.encode(text1, convert_to_tensor=True)
    emb2 = embedder.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return similarity

# === MAIN ===
if __name__ == "__main__":
    old_resume_path = "old_resume.docx"
    new_resume_path = "new_resume.pdf"

    print("Loading and extracting text from resumes...")
    old_text = get_text_from_file(old_resume_path)
    new_text = get_text_from_file(new_resume_path)

    print("\n🔍 Extracting Key Skills (KeyBERT)...")
    old_skills = extract_skills_keybert(old_text)
    new_skills = extract_skills_keybert(new_text)
    print("Old Resume Skills:", old_skills)
    print("New Resume Skills:", new_skills)

    print("\n🧠 Extracting Named Entities (spaCy)...")
    old_entities = extract_entities_spacy(old_text)
    new_entities = extract_entities_spacy(new_text)
    print("Old Entities:", old_entities)
    print("New Entities:", new_entities)

    print("\n📏 Comparing Resume Similarity...")
    similarity_score = compare_similarity(old_text, new_text)
    print(f"Cosine Similarity Score: {similarity_score:.4f}")

    changed_skills = set(new_skills) - set(old_skills)
    print("\n⚙️ Updated/New Skills:")
    print(changed_skills if changed_skills else "No new skills detected.")
