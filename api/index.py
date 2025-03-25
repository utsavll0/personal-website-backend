from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
import dotenv
from groq import Groq
from flask_cors import CORS

dotenv.load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)
CORS(app)

def retrieve(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)

    return [match['metadata'] for match in results['matches']]

def generate_prompt(query, retrieved_projects):
    prompt = f"User Query: {query}\n\n"
    prompt += "Here are some relevant projects:\n"
    for project in retrieved_projects:
        prompt += f"Project Name: {project['name']}\n"
        prompt += f"Description: {project['description']}\n"
        prompt += f"Tags: {', '.join(project['tags'])}\n"
        prompt += f"Type: {', '.join(project['type'])}\n\n"
    prompt += "Answer the user's query based on the above projects."
    return prompt

def generate_response(prompt):
    completion = groq_client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "system", 
                "content": """You are a professional assistant that provides clear, concise responses about projects.
                             Focus on answering the user's query based on the provided project information.
                             Also answer in first person perspective and answer in less than 200 words.                            
                             """
            },
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=32000,
        temperature=0.7,
        stream=False
    )
    
    chat_response = completion.choices[0].message.content
    think_end = chat_response.find("</think>")
    if think_end != -1:
        chat_response = chat_response[think_end + len("</think>"):].strip()
    return chat_response

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_query = data["query"]
        retrieved_projects = retrieve(user_query)
        prompt = generate_prompt(user_query, retrieved_projects)
        response = generate_response(prompt)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

if __name__ == '__main__':
    app.run(debug=False)