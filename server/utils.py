import pandas as pd
import numpy as np
from logistic import Logistic
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score

def clean_data():
    df = pd.read_csv("student-mental-health.csv")

    #Set gender columns to lowercase (I found some male and Male)
    df["Gender"] = df["Gender"].str.lower()
    df["Gender"] = df["gender"].replace({"male": 1, "female": 0})

    # get the avg of the student gpa
    df["CGPA"].replace("", np.nan, inplace=True)
    df["CGPA"].fillna(df["CGPA"].astype(float).mean(), inplace=True)


    df["Marital status"] = df["Marital status"].str.lower()
    df["Marital status"] = df["Marital status"].replace({"yes": 1, "no": 0})


    # get integer based majors

    df["Course"] = df["Course"].str.lower()

    #Hot encode the course column
    df = pd.get_dummies(df, columns=["Course"], prefix="course", drop_first=True)


    # change mental health stats to 1 and 0
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
    df.drop(columns=["Timestamp"], inplace=True)

    # Add a new column called "Mental health risk" that is 1 if any mh columns are 1 and 0 if not
    df["Mental health risk"] = df[[
        "Do you have Depression?", 
        "Do you have Anxiety?", 
        "Do you have Panic attack?", 
        "Did you seek any specialist for a treatment?"
    ]].any(axis=1).replace({True: 1, False: 0})

    return df

def split_data(X, y, test_size=0.2):
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def cross_validation(model: LogisticRegression, X, y, k=5):
    # perform k-fold cross validation
    folds = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in folds.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))

    return np.mean(scores)

def train_model(model: Logistic, X, y):
    # train the model
    X_train, X_test, y_train, y_test = split_data(X, y)
    model.train()

def main():
    df = clean_data()
    X = df.drop(columns=["Mental health risk"])
    y = df["Mental health risk"]
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = Logistic(X_train, y_train, "Mental health risk")
    model.train()
    print(f"Model mean performance: {cross_validation(model.model, X, y)}")

if(__name__ == "__main__"):
    main()