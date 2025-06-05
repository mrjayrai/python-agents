# opportunity_agent.py

import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load dummy consultant skill set data
with open("user_skill_sets_500.json", "r") as f:
    consultants = json.load(f)

# Load opportunity data
with open("opportunities.json", "r") as f:
    opportunities = json.load(f)

texts = [opp["text"] for opp in opportunities]
dates = [opp["date"] for opp in opportunities]

# TF-IDF topic extraction
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# Clustering opportunities
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

# Embed opportunity texts
opportunity_embeddings = model.encode(texts)

# Process consultants
for consultant in consultants:
    skill_names = [skill["name"] for skill in consultant["skills"]]
    skill_text = " ".join(skill_names)

    skill_embedding = model.encode(skill_text)

    scores = cosine_similarity([skill_embedding], opportunity_embeddings)[0]
    matched = [(texts[i], scores[i]) for i in range(len(texts)) if scores[i] > 0.5]

    if matched:
        print(f"\n📋 Consultant ID: {consultant['userId']}")
        print("Matched Opportunities:")
        for opp_text, score in matched:
            print(f" - {opp_text} (Score: {score:.2f})")

# Show clustered opportunities
print("\n📌 Clustered Opportunities:")
for i in range(num_clusters):
    print(f"\nCluster {i + 1}:")
    for j, label in enumerate(clusters):
        if label == i:
            print(f" - {texts[j]} (Date: {dates[j]})")
