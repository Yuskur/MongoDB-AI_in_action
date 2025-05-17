from logistic import Logistic
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
port = 5000

#ONly allow requests from localhost:3000
CORS(app, origins=["http://localhost:3000"])

df = pd.read_csv("student-mental-health.csv")

# ======================================================== DATA CLEANING =======================================================
df[[
    "Do you have Depression?", 
    "Do you have Anxiety?", 
    "Do you have Panic attack?", 
    "Did you seek any specialist for a treatment?"
]] = df[[
    "Do you have Depression?",
    "Do you have Anxiety?", 
    "Do you have Panic attack?", 
    "Did you seek any specialist for a treatment?"]].map({"Yes": 1, "No": 0})

#Set gender columns to lowercase (I found some male and Male)
df["Choose your gender"] = df["Choose your gender"].str.lower()
df["Choose your gender"] = df["Choose your gender"].map({"male": 1, "female": 0})

#Change the name of Choose your gender column to gender for simplicity
df.rename(columns={"Choose your gender", "Gender"}, inplace=True)

df["Your current year of Study"] = df["Your current year of Study"].str.lower()
df["Your current year of Study"] = df["Your current year of Study"].map({
    "year 1": 1,
    "year 2": 2,
    "year 3": 3,
    "year 4": 4,
})

df["What is your CGPA?"] = df["What is your CGPA?"].map({
    ''
})

# We don't really need time stamps for the model so we will remove it
df.drop(columns=["Timestamp"], inplace=True)

# Add a new column called "Mental health risk" that is 1 if any mh columns are 1 and 0 if not
df["Mental health risk"] = df[[
    "Do you have Depression?", 
    "Do you have Anxiety?", 
    "Do you have Panic attack?", 
    "Did you seek any specialist for a treatment?"
]].any(axis=1)

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

# Generate content
response = model.generate_content("List the US bill of rights")
print(response.text)

# ======================================================= API ENDPOINTS =======================================================


# run the server on port 5000
if __name__ == "__main__":
    app.run(debug=True, port=port)