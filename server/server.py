from logistic import Logistic
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
from utils import clean_data, split_data, vectorize_text, generate_text
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import vertexai
from vertexai.language_models import TextEmbeddingModel
import signal
import sys
import os
import logging


load_dotenv()
app = Flask(__name__)
port = 5000

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/yisakor2/Desktop/MongoDB-AI_in_action/server/vertex-ai-keys.json"

#ONly allow requests from localhost:3000
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "http://127.0.0.1:3000"
        ]
    }
}, supports_credentials=True)

#return a cleaned version of the 'student-mental-health.csv' dataset
og_df = pd.read_csv("students_mental_health_survey.csv")
df = clean_data()
print(df.head())

# Initialize the logistic regression model
X = df.drop(columns=["Mental_Health_Risk"])
y = df["Mental_Health_Risk"]
logistic = Logistic(X, y, "Mental_Health_Risk")

# Train the logistic regression model
# logistic.train()
# Train the knn model
# ---------------------- knn train ----------------------


# ======================================================== GEMINI SETUP =======================================================

"""
    Gemini is being used to generate analysis/summary responses of the 
    collection vectors returned from MongoDB vector search
"""
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the model
model = genai.GenerativeModel('gemini-2.0-flash')

# Generate content [This is a test]
# response = model.generate_content("List the US bill of rights")
# print(response.text)

# ======================================================== VERTEX AI SETUP =======================================================

gcp_project_id = os.getenv("GCP_PROJECT_ID")


# Try making a connection to Vertex AI and initializing the text embedding model
try:
    vertexai.init(project=gcp_project_id, location="us-central1")
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
except Exception as e:
    print(f"Error initializing Vertex AI: {e}")
    sys.exit(1)

#Test the embedding model
sample_text = "This is a 24 year old student who is struggling with crippling anxiety and depression!"

for index, row in df.iterrows():
    print(row.to_dict())
    break

# print(vectorize_text(sample_text, task_type="RETRIEVAL_DOCUMENT", title="Student Mental Health Survey", output_dimensionality=384))

# embedding_model_response = model.get_embeddings([sample_text])
# print(f"Embedding for sample text: {embedding_model_response[0].values}")


# ======================================================== MONGODB SETUP =======================================================
"""
    MongoDB is storing the AI generated vector embeddings of the student mental health survey data
"""

uri = f"mongodb+srv://{os.getenv("MONGODB_USER")}:{os.getenv("MONGODB_PASSWORD")}@cluster0.7n0fjh3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = None
embeddings_collection = None

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")

    db = client["mongodb_ai_inAction"]
    embeddings_collection = db["embeddings"]

    # if the colletion doesnt exist, create vector embeddings of every record for mongodb
    if embeddings_collection.count_documents({}) > 0:
        print("Collection exists")
    else:
        print("Creating collection and generating vector embeddings")
        for index, row in og_df.iterrows():
            print("We have ENTERED!!!")
            # Generate the embedding for the row
            embeddings = vectorize_text(
                text=generate_text(row.to_dict()),
                task_type="RETRIEVAL_DOCUMENT",
                title=None, # No need for a title for every row
                output_dimensionality=384,
                model=model
            )

            print(f"Embeddings: {embeddings}")

            
            # Create a document to insert
            document = {
                "record": row.to_json(),
                "text": row.to_json(),
                "embedding": embeddings
            }
            
            print("inserting document")
            # Insert the document into the collection
            embeddings_collection.insert_one(document)
except Exception as e:
    print(f"ERROR: {e}")


# ======================================================= API ENDPOINTS =======================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)

    return

@app.route('/api/query', methods=['POST'])
def query():
    print("Entered Query endpoint")
    body = request.get_json()
    query = body.get('query')

    if not query:
        return jsonify({"error": "Couldn't find query"}), 400

    vectorized_query = vectorize_text(
        text=query,
        task_type="RETRIEVAL_QUERY",
        title=None,
        output_dimensionality=384,
        model=model
    )

    try:
        print("Trying to fetch similar records")
        records = embeddings_collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": vectorized_query,
                    "numCandidates": 1000,
                    "limit": 50
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "record": 1
                }
            }
        ])
        response = list(records)
        return jsonify(response), 200
    except Exception as e:
        logging.exception("MongoDB query failed")
        return jsonify({"error": "Internal Error (MongoDB)"}), 500
    


# run the server on port 5000
if __name__ == "__main__":
    app.run(debug=True, port=port)