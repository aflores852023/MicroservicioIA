import os, time, logging, json

# === Limpieza temprana de variables de entorno (Render inyecta proxies) ===
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    if proxy_var in os.environ:
        print(f"🧹 Eliminando {proxy_var} del entorno Render para evitar bug con openai")
        del os.environ[proxy_var]

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from openai import OpenAI
import httpx

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

# 👇 Cliente OpenAI blindado: ignora completamente las variables de entorno
http_client = httpx.Client(trust_env=False, timeout=30)
client_ai = OpenAI(api_key=OPENAI_KEY, http_client=http_client)
_ready = True


@app.get("/")
def home():
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
            logging.info("🤖 Procesando consulta con OpenAI moderno (Render)")
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
