from logistic import Logistic
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
from utils import clean_data, split_data
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from vertexai.language_models import TextEmbeddingModel
import vertexai
import signal
import sys
import os


load_dotenv()
app = Flask(__name__)
port = 5000

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "server/vertex-ai-keys.json"

#ONly allow requests from localhost:3000
CORS(app, origins=["http://localhost:3000"])

#return a cleaned version of the 'student-mental-health.csv' dataset
df = clean_data()
print(df.head())


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

gcp_project_id = "mongo-challenge-ai"

vertexai.init(project=gcp_project_id, location="us-central1")

embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

#Test the embedding model
sample_text = "This is a 24 year old student who is struggling with crippling anxiety and depression!"

embedding_model_response = embedding_model.get_embeddings([sample_text])
print(f"Embedding for sample text: {embedding_model_response.embeddings[0]}")


# ======================================================== MONGODB SETUP =======================================================
"""
    MongoDB is storing the AI generated vector embeddings of the student mental health survey data
"""

uri = f"mongodb+srv://{os.getenv("MONGODB_USER")}:{os.getenv("MONGODB_PASSWORD")}@cluster0.7n0fjh3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = None
featues_collection = None

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")

    db = client["mongodb_ai_inAction"]
    featues_collection = db["embeddings"]

    # if the colletion doesnt exist, create vector embeddings of every record for mongodb
    if featues_collection:
        print("Collection exists")
    else:
        print("Collection does not exist")

    
except Exception as e:
    print(e)



# Initialize the logistic regression model

X = df.drop(columns=["Mental_Health_Risk"])
y = df["Mental_Health_Risk"]
logistic = Logistic(X, y, "Mental_Health_Risk")


# ======================================================= API ENDPOINTS =======================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()

    df = pd.DataFrame(data)

    return



# run the server on port 5000
if __name__ == "__main__":
    app.run(debug=True, port=port)