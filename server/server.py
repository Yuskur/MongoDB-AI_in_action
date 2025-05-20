from logistic import Logistic
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
from utils import clean_data
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import signal
import sys
import os

load_dotenv()
app = Flask(__name__)
port = 5000

#ONly allow requests from localhost:3000
CORS(app, origins=["http://localhost:3000"])

#return a cleaned version of the 'student-mental-health.csv' dataset
df = clean_data()
print(df.head())

# Initialize the logistic regression model
logistic = Logistic(df, "Mental health risk")
"""
    Initialize the knn model
    ================== KNN class instance =====================
"""

# Train the logistic regression model
# logistic.train()
# Train the knn model
# ---------------------- knn train ----------------------


# ======================================================== GEMINI SETUP =======================================================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the model
model = genai.GenerativeModel('gemini-2.0-flash')

# Generate content [This is a test]
# response = model.generate_content("List the US bill of rights")
# print(response.text)


# ======================================================== MONGODB SETUP =======================================================

uri = f"mongodb+srv://{os.getenv("MONGODB_USER")}:{os.getenv("MONGODB_PASSWORD")}@cluster0.7n0fjh3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# ======================================================= API ENDPOINTS =======================================================




# run the server on port 5000
if __name__ == "__main__":
    app.run(debug=True, port=port)