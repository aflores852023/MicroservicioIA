from flask import Flask, request, jsonify
from pymongo import MongoClient
import os, requests

# === Inicialización Flask ===
app = Flask(__name__)

# === Configuración ===
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "system-stock"
COLLECTION_NAME = "articles"

# === (Opcional) Configurar Ollama local si existe ===
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"

# === Modelo remoto gratuito (backup) ===
HUGGINGFACE_MODEL = "mistralai/Mistral-7B-v0.1"
HUGGINGFACE_URL = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("message", "")

    # === Conectar a MongoDB Atlas ===
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        count = collection.count_documents({})
        sample = collection.find_one()
        context = f"Hay {count} artículos cargados. Ejemplo: {sample.get('name', 'sin nombre')}."
    except Exception as e:
        context = f"No se pudo acceder a MongoDB: {e}"

    # === Si está disponible Ollama local ===
    if USE_OLLAMA:
        try:
            payload = {"model": "mistral", "prompt": f"{context}\n{question}"}
            res = requests.post(OLLAMA_URL, json=payload)
            if res.status_code == 200:
                text = res.json().get("response", "")
                return jsonify({"response": text})
        except Exception as e:
            print("⚠️ Ollama no disponible:", e)

    # === Si Ollama no responde, usar HuggingFace (fallback) ===
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"} if HUGGINGFACE_TOKEN else {}
    res = requests.post(HUGGINGFACE_URL, headers=headers, json={
        "inputs": f"Contexto: {context}\nPregunta: {question}"
    })
    try:
        text = res.json()[0]["generated_text"]
    except Exception:
        text = str(res.json())
    return jsonify({"response": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
