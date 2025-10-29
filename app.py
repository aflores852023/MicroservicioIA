import os, time, logging, json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient

# === CONFIGURACIÓN BASE ===
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "https://systemstock.vercel.app",
    "https://frontend-stock-system-demo.vercel.app",
    "http://localhost:3000"
]}}, supports_credentials=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# === VARIABLES DE ENTORNO ===
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = os.getenv("MONGO_DB", "system-stock")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "articles")

@app.get("/")
def home():
    mode = (
        "ollama" if USE_OLLAMA else
        "openai" if OPENAI_KEY else
        "offline"
    )
    return jsonify({
        "status": "ok",
        "message": f"🤖 Microservicio IA activo (modo: {mode})"
    }), 200


@app.post("/api/query")
def query():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Debe enviar un campo 'message'"}), 400

    try:
        # 🧠 Si está configurado para Ollama, pero no disponible (Render)
        if USE_OLLAMA:
            logging.warning("⚠️ Ollama no está disponible en Render. Cambiando a modo offline.")
            mode = "offline"
        elif OPENAI_KEY:
            # Si existiera API Key (opcional), podrías reactivar esta parte
            from openai import OpenAI
            client_ai = OpenAI(api_key=OPENAI_KEY)
            response = client_ai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": question}],
                temperature=0.3
            )
            answer = response.choices[0].message.content
            return jsonify({"response": answer, "mode": "online"})
        else:
            mode = "offline"

        # === 🧩 MODO OFFLINE ===
        mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
        results = list(
            mongo[DB_NAME][COLLECTION_NAME].find(
                {"name": {"$regex": question, "$options": "i"}}, {"_id": 0}
            )
        )

        if results:
            resp = f"🔍 Encontradas {len(results)} coincidencias en base local."
        elif "hola" in question.lower():
            resp = "👋 ¡Hola! Estoy activo en modo offline, listo para ayudarte."
        else:
            resp = "🤖 Estoy en modo offline. No tengo coincidencias locales para esa búsqueda."

        return jsonify({
            "response": resp,
            "examples": results[:3],
            "mode": mode
        })

    except Exception as e:
        logging.error(f"❌ Error en /api/query: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    logging.info(f"🚀 Iniciando Microservicio IA en puerto {port}")
    app.run(host="0.0.0.0", port=port)
