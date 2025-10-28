from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from llama_index.core import Document, GPTVectorStoreIndex, ServiceContext, set_global_service_context
from llama_index.readers.mongodb import SimpleMongoReader
from llama_index.llms.openai import OpenAI
import os, time, logging, json

# === CONFIG INICIAL ===
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "https://systemstock.vercel.app",
    "https://frontend-stock-system-demo.vercel.app",
    "http://localhost:3000"
]}}, supports_credentials=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context)


MONGO_URI = os.getenv("MONGO_URI", "").strip()
DB_NAME = os.getenv("MONGO_DB", "system-stock")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "articles")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()

_index_cache = None
_ready = False
_offline_mode = False
_last_init_attempt = 0  # evita inicializaciones múltiples


# === CARGA DE DOCUMENTOS ===
def _load_docs_with_reader(mongo_uri: str, db: str, coll: str):
    """Carga documentos desde Mongo, probando distintas versiones de llama_index."""
    reader = SimpleMongoReader(uri=mongo_uri)
    try:
        return reader.load_data(db, coll)
    except TypeError:
        logging.warning("load_data(db, coll) no soportado")
    try:
        return reader.load_data(database_name=db, collection_name=coll)
    except TypeError:
        logging.warning("load_data(database_name=, collection_name=) no soportado")
    try:
        return reader.load_data(db_name=db, collection_name=coll)
    except TypeError:
        logging.warning("load_data(db_name=, collection_name=) no soportado")

    # Fallback manual con PyMongo → Document
    logging.warning("⚠️ Usando fallback manual con PyMongo")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=8000)
    cur = client[db][coll].find({}, {"_id": 0})
    return [Document(text=json.dumps(d, ensure_ascii=False)) for d in cur]


# === INDEXACIÓN ===
def init_index(force=False):
    """Inicializa o reconstruye el índice vectorial."""
    global _index_cache, _ready, _offline_mode, _last_init_attempt

    if not force and time.time() - _last_init_attempt < 30:
        logging.warning("⏳ Esperando antes de reintentar inicialización.")
        return
    _last_init_attempt = time.time()

    if not MONGO_URI:
        raise RuntimeError("MONGO_URI no configurada")

    try:
        logging.info("🔌 Conectando a MongoDB...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
        client.admin.command("ping")
        logging.info("✅ Conexión Mongo OK")

        docs = _load_docs_with_reader(MONGO_URI, DB_NAME, COLLECTION_NAME)
        logging.info(f"📦 {len(docs)} documentos cargados desde {DB_NAME}.{COLLECTION_NAME}")

        if OPENAI_KEY:
            logging.info("🤖 Inicializando índice con OpenAI (modo online)")
            _index_cache = GPTVectorStoreIndex.from_documents(docs)
            _offline_mode = False
        else:
            logging.warning("⚠️ No hay OPENAI_API_KEY — iniciando en modo OFFLINE")
            _index_cache = None
            _offline_mode = True

        _ready = True
        logging.info("🧱 Índice inicializado correctamente")

    except Exception as e:
        _ready = False
        _offline_mode = True
        logging.error(f"⚠️ Fallo inicializando índice: {e}")


def ensure_ready():
    """Verifica si el índice está inicializado."""
    global _ready
    if not _ready:
        logging.warning("⏳ Índice no listo, inicializando…")
        init_index()
    else:
        logging.info("✅ Índice ya está listo")


# === ENDPOINTS ===
@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "🤖 Microservicio IA activo",
        "mode": "offline" if _offline_mode else "online"
    }), 200


@app.get("/healthz")
def healthz():
    return jsonify({
        "ok": _ready,
        "offline_mode": _offline_mode,
        "mongo_uri_configured": bool(MONGO_URI),
        "index_cached": _ready
    }), 200 if _ready else 503


@app.post("/api/query")
def query():
    """Recibe una pregunta y responde usando IA o modo offline."""
    global _index_cache
    start = time.time()

    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Debe enviar un campo 'message'"}), 400

    if not _ready:
        logging.warning("💤 Microservicio IA en standby — devolviendo aviso temporal.")
        return jsonify({
            "response": "🕐 El asistente se está reactivando, intentá nuevamente en unos segundos.",
            "standby": True
        }), 503

    try:
        if not _offline_mode and _index_cache:
            response = _index_cache.as_query_engine().query(question)
            elapsed = round(time.time() - start, 2)
            logging.info(f"✅ Consulta procesada con IA en {elapsed}s")
            return jsonify({
                "response": str(response),
                "elapsed": elapsed,
                "mode": "online"
            })
        else:
            # === MODO OFFLINE ===
            logging.info("🧭 Procesando consulta en modo OFFLINE")
            client = MongoClient(MONGO_URI)
            collection = client[DB_NAME][COLLECTION_NAME]
            results = list(collection.find({"name": {"$regex": question, "$options": "i"}}, {"_id": 0}))
            if results:
                preview = results[:3]  # muestra los primeros
                return jsonify({
                    "response": f"🔍 Encontré {len(results)} coincidencias relacionadas con '{question}'.",
                    "examples": preview,
                    "mode": "offline"
                })
            else:
                return jsonify({
                    "response": f"🤖 No encontré coincidencias locales para '{question}'.",
                    "mode": "offline"
                })

    except Exception as e:
        logging.error(f"❌ Error procesando /api/query: {e}")
        return jsonify({
            "response": "⚠️ No pude procesar tu consulta.",
            "error": str(e)
        }), 500


# === MAIN ===
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    logging.info(f"🚀 Iniciando Microservicio IA en puerto {port}")
    try:
        init_index()
    except Exception as e:
        logging.error(f"⚠️ Fallo inicializando índice: {e}")
    app.run(host="0.0.0.0", port=port)
