from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from llama_index.readers.mongodb import SimpleMongoReader
from llama_index import GPTVectorStoreIndex  # ok con 0.9.35
import os, logging

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "https://systemstock.vercel.app",
    "https://frontend-stock-system-demo.vercel.app",
    "http://localhost:3000",
]}}, supports_credentials=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MONGO_URI = (os.getenv("MONGO_URI") or "").strip()
DB_NAME = os.getenv("MONGO_DB", "system-stock")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "articles")

_mongo = None
_index = None

def ensure_ready():
    global _mongo, _index
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI no configurada en Render > Environment")

    if _mongo is None:
        logging.info("🔌 Conectando a Mongo…")
        _mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
        _mongo.admin.command("ping")
        logging.info("✅ Ping Mongo OK")

    if _index is None:
        reader = SimpleMongoReader(_mongo)
        docs = reader.load_data(database_name=DB_NAME, collection_name=COLLECTION_NAME)
        logging.info(f"📦 Cargados {len(docs)} docs de {DB_NAME}.{COLLECTION_NAME}")
        _index = GPTVectorStoreIndex.from_documents(docs)
        logging.info("🧱 Índice vectorial listo")

@app.get("/")
def home():
    return jsonify({"status": "ok", "message": "🤖 Microservicio IA activo"}), 200

@app.get("/healthz")
def healthz():
    try:
        if not MONGO_URI:
            return jsonify({"ok": False, "error": "MONGO_URI missing"}), 500
        _c = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        _c.admin.command("ping")
        return jsonify({"ok": True}), 200
    except Exception as e:
        logging.error(f"healthz error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/api/query")
def query():
    try:
        data = request.get_json(silent=True) or {}
        q = (data.get("message") or "").strip()
        if not q:
            return jsonify({"error": "Debe enviar un campo 'message'"}), 400

        ensure_ready()
        resp = _index.as_query_engine().query(q)
        return jsonify({"response": str(resp)})

    except RuntimeError as re:
        logging.error(f"❌ Config error: {re}")
        return jsonify({"response": "⚠️ Config faltante.", "error": str(re)}), 500
    except Exception as e:
        logging.error(f"❌ Error /api/query: {e}")
        return jsonify({"response": "⚠️ No pude procesar tu consulta.", "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logging.info(f"🚀 Iniciando Microservicio IA en puerto {port}")
    app.run(host="0.0.0.0", port=port)
