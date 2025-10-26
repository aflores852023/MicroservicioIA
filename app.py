from flask import Flask, request, jsonify
from pymongo import MongoClient
from llama_index import SimpleMongoReader, VectorStoreIndex
# from llama_index.llms.ollama import Ollama   # ⚠️ activalo solo si usás Ollama local
import os

app = Flask(__name__)

# === Configuración de Mongo ===
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "system-stock"
COLLECTION_NAME = "articles"

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "🤖 Microservicio IA activo y escuchando"}), 200


@app.route("/api/query", methods=["POST"])
def query():
    try:
        data = request.get_json(silent=True)
        if not data or "message" not in data:
            return jsonify({"error": "Debe enviar un campo 'message'"}), 400

        question = data["message"]

        # Conectar a Mongo
        client = MongoClient(MONGO_URI)
        reader = SimpleMongoReader(client)
        docs = reader.load_data(database_name=DB_NAME, collection_name=COLLECTION_NAME)

        # Crear índice temporal
        index = VectorStoreIndex.from_documents(docs)
        query_engine = index.as_query_engine()

        response = query_engine.query(question)
        return jsonify({"response": str(response)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
