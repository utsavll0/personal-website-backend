from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pinecone import Pinecone, ServerlessSpec
import dotenv
import os

dotenv.load_dotenv()

index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Create the Pinecone index (if not already created)
# pc.create_index(index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

# Load project data
with open("resources/projects.json", "r") as f:
    projects = json.load(f)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to format project data for embedding
def format_project_text(project):
    """Convert project fields into a single text representation."""
    name = project["name"]
    description = project["description"]
    tags = " ".join(tag["label"] for tag in project.get("tags", []))
    types = " ".join(project.get("type", [])) 

    return f"{name}. {description}. Tags: {tags}. Type: {types}."

# Generate embeddings
project_texts = [format_project_text(proj) for proj in projects]
project_embeddings = model.encode(project_texts, convert_to_numpy=True)

# Initialize Pinecone index
index = pc.Index(index_name)

# Prepare metadata and upsert embeddings into Pinecone
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

# Upsert the data into Pinecone
index.upsert(vectors=upsert_data)

# Save metadata for retrieval (you can retrieve this later if needed)
with open("resources/projects_metadata.json", "w") as f:
    json.dump(projects, f, indent=2)

print("Embeddings stored successfully in Pinecone!")
