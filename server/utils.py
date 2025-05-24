import pandas as pd
import numpy as np
from logistic import Logistic
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score

def clean_data():
    df = pd.read_csv("students_mental_health_survey.csv")

    #Set gender columns to lowercase (I found some male and Male)
    df["Gender"] = df["Gender"].str.lower()
    df["Gender"] = df["Gender"].replace({"male": 1, "female": 0})

    # get the avg of the student gpa and fill the empty values with it
    df["CGPA"].replace("", np.nan, inplace=True)
    df["CGPA"].fillna(df["CGPA"].astype(float).mean(), inplace=True)

    df["Relationship_Status"] = df["Relationship_Status"].str.lower()
    df["Relationship_Status"] = df["Relationship_Status"].replace({"yes": 1, "no": 0})


    # get integer based majors
    #Hot encode the course column
    df["Course"] = df["Course"].str.lower()
    df = pd.get_dummies(df, columns=["Course"], prefix="course", drop_first=False)

    #Hot encode the sleep quality column
    df["Sleep_Quality"] = df["Sleep_Quality"].str.lower()
    df = pd.get_dummies(df, columns=["Sleep_Quality"], prefix="sleep_quality", drop_first=False)

    #Hot encode the Pysical activity column
    df["Physical_Activity"] = df["Physical_Activity"].str.lower()
    df = pd.get_dummies(df, columns=["Physical_Activity"], prefix="physical_activity", drop_first=False)

    #Hot encode the Diet quality column
    df["Diet_Quality"] = df["Diet_Quality"].str.lower()
    df = pd.get_dummies(df, columns=["Diet_Quality"], prefix="diet_quality", drop_first=False)

    #Hot encode the Soclial support column
    df["Social_Support"] = df["Social_Support"].str.lower()
    df = pd.get_dummies(df, columns=["Social_Support"], prefix="social_support", drop_first=False)

    #Hot encode the Relationship status column
    df["Relationship_Status"] = df["Relationship_Status"].str.lower()
    df = pd.get_dummies(df, columns=["Relationship_Status"], prefix="relationship_status", drop_first=False)

    #Hot encode the Substance use column
    df["Substance_Use"] = df["Substance_Use"].str.lower()
    df = pd.get_dummies(df, columns=["Substance_Use"], prefix="substance_use", drop_first=False)

    #Hot encode the Counseling service use column
    df["Counseling_Service_Use"] = df["Counseling_Service_Use"].str.lower()
    df = pd.get_dummies(df, columns=["Counseling_Service_Use"], prefix="counseling_service_use", drop_first=False)

    #Map the Family history column to 1 and 0
    df["Family_History"] = df["Family_History"].str.lower()
    df["Family_History"] = df["Family_History"].replace({"yes": 1, "no": 0})

    #Map Choronic illness to 1 and 0
    df["Chronic_Illness"] = df["Chronic_Illness"].str.lower()
    df["Chronic_Illness"] = df["Chronic_Illness"].replace({"yes": 1, "no": 0})

    #Hot encode Extracurricular Involvement column
    df["Extracurricular_Involvement"] = df["Extracurricular_Involvement"].str.lower()
    df = pd.get_dummies(df, columns=["Extracurricular_Involvement"], prefix="extracurricular_involvement", drop_first=False)

    #Hot encode Residence type column
    df["Residence_Type"] = df["Residence_Type"].str.lower()
    df = pd.get_dummies(df, columns=["Residence_Type"], prefix="residence_type", drop_first=False)

    """
        We consider a student to be at risk of mental health issues if they have a score of 3 or more in any one of the following categories:
        - Stress Score
        - Depression Score
        - Anxiety Score
    """

    #Mental health label
    stress_label = df["Stress_Level"] >= 3
    depression_label = df["Depression_Score"] >= 3
    anxiety_label = df["Anxiety_Score"] >= 3

    df["Mental_Health_Risk"] = (stress_label | depression_label | anxiety_label).astype(int)
    
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

# We evaluate our models here using cross validation
def main():
    df = clean_data()
    X = df.drop(columns=["Mental_Health_Risk"])
    y = df["Mental_Health_Risk"]
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = Logistic(X_train, y_train, "Mental_Health_Risk")
    model.train()
    print(f"\n|\n|\n|\nv\nModel mean performance: {cross_validation(model.model, X, y)}")

if(__name__ == "__main__"):
    main()