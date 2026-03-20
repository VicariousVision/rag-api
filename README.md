# rag-api

This is a small project demonstarating how to build a RAG based on ollama, qwen2.5:0.5b, chromaDB, FastAPI and nombic-embed-text.

Clone the repo, install these:

1. install ollama
2. ollama pull qwen2.5:0.5b
3. ollama run qwen2.5:0.5b
4. pip install fastapi uvicorn chromadb ollama
5. ollama pull nomic-embed-text

Verify the model installations:
ollama list


Run these:
python build_knowledge_base.py
uvicorn main:app --reload

Open your browser and go to http://127.0.0.1:8000/docs to use Swagger to send the model questions based on your profile. 
