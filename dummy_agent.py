# generate_dummy_data.py

import random
import uuid
import json
from faker import Faker
from bson import ObjectId

fake = Faker()

skills_pool = [
    "Python", "JavaScript", "Java", "C#", "Go", "Rust", "SQL", "HTML", "CSS", "React",
    "Node.js", "MongoDB", "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Django", "Flask", "TensorFlow"
]

certifications = [
    "AWS Certified Developer", "Certified Kubernetes Admin", "MongoDB Certified", "Google Data Engineer",
    None, None, None  # Increase chance of no certification
]

def generate_skill():
    return {
        "name": random.choice(skills_pool),
        "yearsOfExperience": random.randint(1, 10),
        "certification": random.choice(certifications),
        "endorsements": random.randint(1, 20)
    }

def generate_project():
    num_skills = random.randint(1, 4)
    return {
        "githubUrl": fake.url(),
        "projectInfo": fake.text(max_nb_chars=100),
        "skillsUsed": random.sample(skills_pool, num_skills),
        "timeConsumedInDays": random.randint(7, 180)
    }

def generate_user_skill_set():
    return {
        "userId": str(ObjectId()),
        "skills": [generate_skill() for _ in range(random.randint(2, 6))],
        "projects": [generate_project() for _ in range(random.randint(1, 3))]
    }

# Generate 500 documents
data = [generate_user_skill_set() for _ in range(500)]

# Save to file
with open("user_skill_sets_500.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ 500 dummy UserSkillSet documents saved to 'user_skill_sets_500.json'")
