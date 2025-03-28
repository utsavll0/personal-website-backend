---
# LLM Chatbot API
---

Welcome to the LLM Chatbot API! This project is designed to integrate an intelligent chatbot into my personal website, enabling users to get insightful responses based on pre-stored data.

## Tech Stack

The following technologies power the LLM Chatbot API:

- **Flask**: Lightweight web framework to expose the API endpoints.
- **Sentence Transformers**: Used for generating embeddings from textual data.
- **Pinecone**: A vector database to store and efficiently retrieve sentence embeddings.
- **Groq (Not to be confused with Grok by Elon Musk!)**: A Large Language Model (LLM) used for generating insightful responses.

## How It Works

Here’s a step-by-step breakdown of the process:

### 1. **Data Preparation**

- All input data is stored in a structured JSON format. This data serves as the knowledge base for the chatbot.

### 2. **Embeddings Creation**

- The data is processed through the **all-MiniLM-L6-v2** model (part of the Sentence Transformers library) to convert each sentence into high-dimensional embeddings.

### 3. **Storing Embeddings**

- The generated embeddings are stored in **Pinecone**, a vector database, allowing for efficient and fast retrieval during query processing.

### 4. **Retrieving Relevant Results**

- When a user sends a query, the API retrieves the top 5 most relevant results from the vector database using **cosine similarity**. This ensures the most relevant pieces of information are returned.

### 5. **Final Response Generation**

- The top 5 results are then passed to the **deepseek-r1-distill-llama-70b** Large Language Model (LLM) to generate a coherent and contextually appropriate response to the user’s query.

---

## Features

- **Fast Response Time**: The use of vector embeddings ensures efficient retrieval of relevant data.
- **Seamless Integration**: Easily integrates with your personal website for a more interactive experience.
- **Scalable**: Can handle large datasets and provide insightful responses across various domains.

## Running the Project

### Prerequisites

- Python 3.x
- Install dependencies via `pip install -r requirements.txt`.

### Running the embeddings setup

1. Change directory to `embeddings/`
2. Run `python embeddings.py` to run the embeddings code

This will transform all data in the json file to embeddings and store in the pinecone db.

### Running the API

1. Clone this repository.
2. Install the required packages.
3. Run the Flask app:
   ```bash
   python api/index.py
   ```

The API will be up and running, ready to handle your queries!

---

## Conclusion

This API serves as a powerful tool for integrating AI-driven conversations into any platform, allowing for dynamic interactions based on your custom dataset. With the combination of vector embeddings and an LLM, this project provides an intelligent and scalable solution for creating a chatbot that can understand and respond with meaningful information.
