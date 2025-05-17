from logistic import Logistic
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
port = 5000

#ONly allow requests from localhost:3000
CORS(app, origins=["http://localhost:3000"])

df = pd.read_csv("student-mental-health.csv")

# Clean the data replacing values like yes and no with 1 and 0
df[[
    "Do you have Depression?", 
    "Do you have Anxiety?", 
    "Do you have Panic attack?", 
    "Did you seek any specialist for a treatment?"
]] = df[[
    "Do you have Depression?",
    "Do you have Anxiety?", 
    "Do you have Panic attack?", 
    "Did you seek any specialist for a treatment?"]].replace({"Yes": 1, "No": 0})

# We don't really need time stamps for the model so we will remove it
df.drop(columns=["Timestamp"])

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
logistic.train()
# Train the knn model
# ---------------------- knn train ----------------------


# run the server on port 5000
if __name__ == "__main__":
    app.run(debug=True, port=port)