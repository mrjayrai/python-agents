# training_agent.py

import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load user skill set data (same as previously generated file)
with open("user_skill_sets_500.json", "r") as f:
    consultants = json.load(f)

# Simulated required skills for each consultant (in production, this would come from MongoDB)
required_skills_map = {
    consultant["userId"]: ["AWS", "Docker", "Python", "React"]
    for consultant in consultants
}

# Simulated completed trainings for each consultant (replace with real MongoDB collection)
completed_trainings_map = {
    consultant["userId"]: ["AWS", "Kubernetes", "Flask", "SQL"]
    for consultant in consultants
}

def embed_skills(skills):
    return model.encode([" ".join(skills)])[0]

report_data = []

for consultant in consultants:
    user_id = consultant["userId"]
    current_skills = [s["name"] for s in consultant["skills"]]
    required_skills = required_skills_map.get(user_id, [])
    trainings = completed_trainings_map.get(user_id, [])

    # Embeddings
    required_vec = embed_skills(required_skills)
    trained_vec = embed_skills(trainings)
    current_vec = embed_skills(current_skills)

    trained_score = cosine_similarity([trained_vec], [required_vec])[0][0]
    current_score = cosine_similarity([current_vec], [required_vec])[0][0]

    report_data.append({
        "userId": user_id,
        "trained_vs_required": round(trained_score, 2),
        "current_vs_required": round(current_score, 2),
        "required_skills": required_skills,
        "trainings": trainings,
        "gaps": list(set(required_skills) - set(trainings) - set(current_skills))
    })

# Plot sample visual for first consultant
first = report_data[0]
labels = ["Training Progress", "Current Skill Alignment"]
values = [first["trained_vs_required"], first["current_vs_required"]]

# plt.bar(labels, values, color=["skyblue", "green"])
# plt.title(f"Skill Growth Report for Consultant {first['userId'][:6]}...")
# plt.ylim(0, 1)
# plt.ylabel("Cosine Similarity")
# plt.tight_layout()
# plt.savefig("training_progress_sample.png")
# plt.show()

# Print summary for top 3 consultants
for i, report in enumerate(report_data[:1]):
    print(f"\n📋 Consultant {i+1} ({report['userId']}):")
    print(f" - Trained vs Required: {report['trained_vs_required']}")
    print(f" - Current vs Required: {report['current_vs_required']}")
    print(f" - Skill Gaps: {report['gaps']}")
