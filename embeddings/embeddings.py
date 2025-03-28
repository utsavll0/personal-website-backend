from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pinecone import Pinecone, ServerlessSpec
import dotenv
import os

dotenv.load_dotenv()

index_name = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

with open("resources/projects.json", "r") as f:
    projects = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

def format_project_text(project):
    """Convert project fields into a single text representation."""
    name = project["name"]
    description = project["description"]
    tags = " ".join(tag["label"] for tag in project.get("tags", []))
    types = " ".join(project.get("type", [])) 

    return f"{name}. {description}. Tags: {tags}. Type: {types}."

project_texts = [format_project_text(proj) for proj in projects]
project_embeddings = model.encode(project_texts, convert_to_numpy=True)

index = pc.Index(index_name)

upsert_data = []
for i, embedding in enumerate(project_embeddings):
    project = projects[i]
    metadata = {
        "name": project["name"],
        "description": project["description"],
        "tags": [tag["label"] for tag in project.get("tags", [])],  # Extract only the 'label' from each tag
        "type": project.get("type", [])
    }
    upsert_data.append((str(i), embedding.tolist(), metadata))  # 'i' as ID for each project

index.upsert(vectors=upsert_data)

with open("resources/projects_metadata.json", "w") as f:
    json.dump(projects, f, indent=2)

print("Embeddings stored successfully in Pinecone!")
