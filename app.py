from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from openai import OpenAI  # ✅ nuevo SDK oficial
import os, time, logging, json

# === CONFIG INICIAL ===
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "https://systemstock.vercel.app",
    "https://frontend-stock-system-demo.vercel.app",
    "http://localhost:3000"
]}}, supports_credentials=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MONGO_URI = os.getenv("MONGO_URI", "").strip()
DB_NAME = os.getenv("MONGO_DB", "system-stock")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "articles")

# === Limpieza de variables de proxy del entorno (Render las define por defecto) ===
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    if proxy_var in os.environ:
        logging.warning(f"🧹 Eliminando variable de entorno {proxy_var} para evitar conflicto con OpenAI SDK")
        del os.environ[proxy_var]

# 👇 Inicializa cliente OpenAI moderno
client_ai = OpenAI(api_key=OPENAI_KEY)
_ready = True


@app.get("/")
def home():
    """Endpoint base de salud."""
    return jsonify({
        "status": "ok",
        "message": "🤖 Microservicio IA activo",
        "mode": "online" if OPENAI_KEY else "offline"
    }), 200


@app.post("/api/query")
def query():
    """Procesa consultas vía OpenAI o búsqueda Mongo."""
    start = time.time()
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()

    if not question:
        return jsonify({"error": "Debe enviar un campo 'message'"}), 400

    try:
        if OPENAI_KEY:
            logging.info("🤖 Procesando consulta con OpenAI moderno")
            response = client_ai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": question}],
                temperature=0.3,
            )
            answer = response.choices[0].message.content
            elapsed = round(time.time() - start, 2)
            return jsonify({
                "response": answer,
                "elapsed": elapsed,
                "mode": "online"
            })
        else:
            # fallback: búsqueda Mongo
            mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
            results = list(
                mongo[DB_NAME][COLLECTION_NAME].find(
                    {"name": {"$regex": question, "$options": "i"}},
                    {"_id": 0}
                )
            )
            return jsonify({
                "response": f"🔍 {len(results)} coincidencias locales para '{question}'.",
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
