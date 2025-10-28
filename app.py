from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from llama_index.core.schema import Document
from llama_index.readers.mongodb import SimpleMongoReader
from llama_index import GPTVectorStoreIndex
import os, time, logging, json

# === CONFIG INICIAL ===
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "https://systemstock.vercel.app",
    "https://frontend-stock-system-demo.vercel.app",
    "http://localhost:3000"
]}}, supports_credentials=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MONGO_URI = os.getenv("MONGO_URI", "").strip()
DB_NAME = os.getenv("MONGO_DB", "system-stock")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "articles")

_index_cache = None
_ready = False
_last_init_attempt = 0  # ⏱️ evita inicializaciones múltiples seguidas


# === CARGA DE DOCUMENTOS ===
def _load_docs_with_reader(mongo_uri: str, db: str, coll: str):
    """Prueba varias firmas de load_data() y cae a PyMongo si es necesario."""
    reader = SimpleMongoReader(uri=mongo_uri)

    try:
        return reader.load_data(db, coll)
    except TypeError:
        logging.warning("load_data(db, coll) no soportado en esta versión")

    try:
        return reader.load_data(database_name=db, collection_name=coll)
    except TypeError:
        logging.warning("load_data(database_name=, collection_name=) no soportado")

    try:
        return reader.load_data(db_name=db, collection_name=coll)
    except TypeError:
        logging.warning("load_data(db_name=, collection_name=) no soportado")

    # 🔸 Fallback manual con PyMongo
    logging.warning("⚠️ Usando fallback manual con PyMongo")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    cur = client[db][coll].find({}, {"_id": 0})
    docs = [Document(text=json.dumps(d, ensure_ascii=False)) for d in cur]
    return docs


# === INDEXACIÓN ===
def init_index(force=False):
    """Inicializa o reconstruye el índice vectorial."""
    global _index_cache, _ready, _last_init_attempt

    # Previene reintentos constantes si Render está despertando
    if not force and time.time() - _last_init_attempt < 30:
        logging.warning("⏳ Esperando antes de reintentar la inicialización del índice.")
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

        _index_cache = GPTVectorStoreIndex.from_documents(docs)
        _ready = True
        logging.info("🧱 Índice vectorial inicializado correctamente")

    except Exception as e:
        _ready = False
        logging.error(f"⚠️ Fallo inicializando índice: {e}")


def ensure_ready():
    """Verifica que el índice esté disponible; si Render está “despertando”, evita errores."""
    global _ready
    if not _ready:
        logging.warning("⏳ Índice no listo, inicializando…")
        init_index()
    else:
        logging.info("✅ Índice ya está inicializado y listo para consultas")


# === ENDPOINTS ===
@app.get("/")
def home():
    return jsonify({"status": "ok", "message": "🤖 Microservicio IA activo"}), 200


@app.get("/healthz")
def healthz():
    """Endpoint para chequeo automático de Render o monitor."""
    return jsonify({
        "ok": _ready,
        "mongo_uri_configured": bool(MONGO_URI),
        "index_cached": _ready
    }), 200 if _ready else 503


@app.post("/api/query")
def query():
    """Recibe una pregunta y responde con el índice vectorial."""
    global _index_cache
    start = time.time()

    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("message") or "").strip()
        if not question:
            return jsonify({"error": "Debe enviar un campo 'message'"}), 400

        if not _ready or _index_cache is None:
            logging.warning("⏳ Modo standby detectado: reactivando índice...")
            init_index()

            # Si todavía no está listo → mensaje temporal
            if not _ready:
                logging.warning("🕐 Render en standby — devolviendo respuesta temporal.")
                return jsonify({
                    "response": "🤖 El asistente se está despertando... intentá nuevamente en unos segundos.",
                    "standby": True
                }), 503

        response = _index_cache.as_query_engine().query(question)
        elapsed = round(time.time() - start, 2)
        logging.info(f"✅ Consulta procesada en {elapsed}s")

        return jsonify({
            "response": str(response),
            "elapsed": elapsed
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
