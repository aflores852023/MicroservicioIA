from flask import Flask, request, jsonify
from pymongo import MongoClient
from flask_cors import CORS
from llama_index.readers.mongodb import SimpleMongoReader
from llama_index import GPTVectorStoreIndex


import os
import logging

# === CONFIGURACIÓN INICIAL ===
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "https://systemstock.vercel.app",
    "https://frontend-stock-system-demo.vercel.app",
    "http://localhost:3000"
]}}, supports_credentials=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# === CONFIG DE MONGO ===
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "system-stock"
COLLECTION_NAME = "articles"

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "🤖 Microservicio IA activo y escuchando"
    }), 200


# === ENDPOINT PRINCIPAL /api/query ===
@app.route("/api/query", methods=["POST"])
def query():
    try:
        data = request.get_json(silent=True)
        if not data or "message" not in data:
            return jsonify({"error": "Debe enviar un campo 'message'"}), 400

        question = data["message"]
        logging.info(f"🧠 Recibida consulta: {question}")

        # === Conectar a Mongo ===
        client = MongoClient(MONGO_URI)
        reader = SimpleMongoReader(client)
        docs = reader.load_data(database_name=DB_NAME, collection_name=COLLECTION_NAME)
        logging.info(f"📦 {len(docs)} documentos cargados desde MongoDB")

        # === Crear índice temporal (GPTVectorStoreIndex) ===
        index = GPTVectorStoreIndex.from_documents(docs)

        query_engine = index.as_query_engine()

        response = query_engine.query(question)
        logging.info(f"✅ Respuesta generada: {response}")

        return jsonify({"response": str(response)})

    except Exception as e:
        logging.error(f"❌ Error procesando /api/query: {e}")
        # Fallback genérico
        return jsonify({
            "response": "⚠️ No pude procesar tu consulta en este momento. Intentá más tarde.",
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logging.info(f"🚀 Iniciando Microservicio IA en puerto {port}")
    app.run(host="0.0.0.0", port=port)
