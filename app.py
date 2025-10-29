import os, time, logging, json, requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "https://systemstock.vercel.app",
    "https://frontend-stock-system-demo.vercel.app",
    "http://localhost:3000"
]}}, supports_credentials=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"  # 👈 por defecto true
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = os.getenv("MONGO_DB", "system-stock")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "articles")

@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "🤖 Microservicio IA activo",
        "mode": "ollama" if USE_OLLAMA else "openai"
    }), 200

@app.post("/api/query")
def query():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Debe enviar un campo 'message'"}), 400

    try:
        if USE_OLLAMA:
            logging.info(f"🧠 Usando Ollama local con modelo '{OLLAMA_MODEL}'")
            r = requests.post("http://localhost:11434/api/generate",
                              json={"model": OLLAMA_MODEL, "prompt": question})
            if r.status_code == 200:
                text = r.json().get("response", "")
                return jsonify({"response": text, "mode": "ollama"})
            else:
                raise Exception(f"Ollama error: {r.text}")

        elif OPENAI_KEY:
            from openai import OpenAI
            client_ai = OpenAI(api_key=OPENAI_KEY)
            response = client_ai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": question}],
                temperature=0.3,
            )
            answer = response.choices[0].message.content
            return jsonify({"response": answer, "mode": "online"})

        else:
            # fallback: búsqueda Mongo
            mongo = MongoClient(MONGO_URI)
            results = list(mongo[DB_NAME][COLLECTION_NAME]
                           .find({"name": {"$regex": question, "$options": "i"}}, {"_id": 0}))
            return jsonify({
                "response": f"🔍 {len(results)} coincidencias locales.",
                "examples": results[:3],
                "mode": "offline"
            })

    except Exception as e:
        logging.error(f"❌ Error en /api/query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    logging.info(f"🚀 Iniciando Microservicio IA en puerto {port}")
    app.run(host="0.0.0.0", port=port)
