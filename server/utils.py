import pandas as pd
import numpy as np
from logistic import Logistic
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, classification_report
from vertexai.language_models import TextEmbeddingInput


# ========================================================= Data Cleaning =======================================================
def clean_data():
    df = pd.read_csv("students_mental_health_survey.csv")

    df.columns = df.columns.str.lower()

    # Set gender column to lowercase and map to 0/1
    df["gender"] = df["gender"].str.lower()
    df["gender"] = df["gender"].replace({"male": 1, "female": 0})

    # Fill missing CGPA values with the mean
    df["cgpa"].replace("", np.nan, inplace=True)
    df["cgpa"].fillna(df["cgpa"].astype(float).mean(), inplace=True)

    df["relationship_status"] = df["relationship_status"].str.lower()
    df["relationship_status"] = df["relationship_status"].replace({"yes": 1, "no": 0})

    # One-hot encode various fields
    df["course"] = df["course"].str.lower()
    df = pd.get_dummies(df, columns=["course"], prefix="course", drop_first=False)

    df["sleep_quality"] = df["sleep_quality"].str.lower()
    df = pd.get_dummies(df, columns=["sleep_quality"], prefix="sleep_quality", drop_first=False)

    df["physical_activity"] = df["physical_activity"].str.lower()
    df = pd.get_dummies(df, columns=["physical_activity"], prefix="physical_activity", drop_first=False)

    df["diet_quality"] = df["diet_quality"].str.lower()
    df = pd.get_dummies(df, columns=["diet_quality"], prefix="diet_quality", drop_first=False)

    df["social_support"] = df["social_support"].str.lower()
    df = pd.get_dummies(df, columns=["social_support"], prefix="social_support", drop_first=False)

    df["relationship_status"] = df["relationship_status"].str.lower()
    df = pd.get_dummies(df, columns=["relationship_status"], prefix="relationship_status", drop_first=False)

    df["substance_use"] = df["substance_use"].str.lower()
    df = pd.get_dummies(df, columns=["substance_use"], prefix="substance_use", drop_first=False)

    df["counseling_service_use"] = df["counseling_service_use"].str.lower()
    df = pd.get_dummies(df, columns=["counseling_service_use"], prefix="counseling_service_use", drop_first=False)

    df["family_history"] = df["family_history"].str.lower()
    df["family_history"] = df["family_history"].replace({"yes": 1, "no": 0})

    df["chronic_illness"] = df["chronic_illness"].str.lower()
    df["chronic_illness"] = df["chronic_illness"].replace({"yes": 1, "no": 0})

    df["extracurricular_involvement"] = df["extracurricular_involvement"].str.lower()
    df = pd.get_dummies(df, columns=["extracurricular_involvement"], prefix="extracurricular_involvement", drop_first=False)

    df["residence_type"] = df["residence_type"].str.lower()
    df = pd.get_dummies(df, columns=["residence_type"], prefix="residence_type", drop_first=False)

    # Mental health risk label based on score thresholds
    stress_label = df["stress_level"] >= 3
    depression_label = df["depression_score"] >= 3
    anxiety_label = df["anxiety_score"] >= 3

    df["mental_health_risk"] = (stress_label | depression_label | anxiety_label).astype(int)

    df = df.drop(columns=["stress_level", "depression_score", "anxiety_score"])

    return df

# ========================================================= Model =======================================================
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

def predict(model: Logistic, new_data: pd.DataFrame):
    # predict the target variable for new data
    print("Predicting new data...")
    return model.predict(new_data)

# ========================================================== Utility Functions =======================================================

def vectorize_text(text, task_type, model, title=None, output_dimensionality=384):

    text_embedding_input = TextEmbeddingInput(
        task_type=task_type, title=title, text=text
    )

    kwargs = (
        dict(output_dimensionality=output_dimensionality)
        if output_dimensionality
        else {}
    )

    try:

        embeddings = model.get_embeddings([text_embedding_input], **kwargs)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

    return embeddings[0].values

"""
    Take a row of the DataFrame as json format and generate a text description of the student for text vectorization model.
"""
def generate_text(row):
    text = f"""
        This student is {row["Age"]} years old.
        This student is {row["Gender"]}.
        This student has a CGPA of {row["CGPA"]}.
        This student has a family history of mental health issues: {row["Family_History"]}.
        This student has a chronic illness: {row["Chronic_Illness"]}.
        This student has a stress level of {row["Stress_Level"]}.
        This student has a depression score of {row["Depression_Score"]}.
        This student has an anxiety score of {row["Anxiety_Score"]}.
        This student is in a relationship: {row["Relationship_Status"]}.
        This student has social support: {row["Social_Support"]}.
        This student has a physical activity level of {row["Physical_Activity"]}.
        This student has a diet quality of {row["Diet_Quality"]}.
        This student has a sleep quality of {row["Sleep_Quality"]}.
        This student has a substance use level of {row["Substance_Use"]}.
        This student has used counseling services: {row["Counseling_Service_Use"]}.
        This student is involved in extracurricular activities: {row["Extracurricular_Involvement"]}.
        This student lives in a {row["Residence_Type"]} residence.
        """
    return text

def get_model_df():
    columns = ['age', 'gender', 'cgpa', 'family_history', 'chronic_illness',
       'financial_stress', 'semester_credit_load', 'course_business',
       'course_computer science', 'course_engineering', 'course_law',
       'course_medical', 'course_others', 'sleep_quality_average',
       'sleep_quality_good', 'sleep_quality_poor', 'physical_activity_high',
       'physical_activity_low', 'physical_activity_moderate',
       'diet_quality_average', 'diet_quality_good', 'diet_quality_poor',
       'social_support_high', 'social_support_low', 'social_support_moderate',
       'relationship_status_in a relationship', 'relationship_status_married',
       'relationship_status_single', 'substance_use_frequently',
       'substance_use_never', 'substance_use_occasionally',
       'counseling_service_use_frequently', 'counseling_service_use_never',
       'counseling_service_use_occasionally',
       'extracurricular_involvement_high', 'extracurricular_involvement_low',
       'extracurricular_involvement_moderate', 'residence_type_off-campus',
       'residence_type_on-campus', 'residence_type_with family',
       ]
    df = pd.DataFrame([[0] * len(columns)], columns=columns)

    return df


def fix_columns(df: pd.DataFrame, new_data: dict):
    df.columns = df.columns.str.lower()

    # Preserve original case for numeric fields
    normalized_data = {}
    for k, v in new_data.items():
        if isinstance(v, str):
            normalized_data[k.lower()] = v.lower()
        else:
            normalized_data[k.lower()] = v 

    for col in df.columns:
        if "_" in col:
            prefix = col[:col.rfind("_")]
            value = col[col.rfind("_")+1:]

            if prefix in normalized_data and isinstance(normalized_data[prefix], str):
                if normalized_data[prefix] == value:
                    df.at[0, col] = 1
                else:
                    df.at[0, col] = 0
        else:
            if col in normalized_data:
                val = normalized_data[col]
                if isinstance(val, str):
                    if val in {"yes", "male"}:
                        df.at[0, col] = 1
                    elif val in {"no", "female"}:
                        df.at[0, col] = 0
                    else:
                        try:
                            df.at[0, col] = float(val)
                        except ValueError:
                            df.at[0, col] = 0
                else:
                    df.at[0, col] = val 

    return df




# =========================================================== Main File Function =======================================================
# We evaluate our models here using cross validation
def main():
    df = clean_data()
    X = df.drop(columns=["Mental_Health_Risk"])
    y = df["Mental_Health_Risk"]
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    logistic = Logistic(X_train, y_train, "mental_health_risk")
    logistic.train()

    y_pred = logistic.model.predict(X_test)

    print("|\n|\n|\n|\n|\n|\n|\nv")
    print(f"Cross-validation score: {cross_validation(logistic.model, X_train, y_train)}")
    print(f"Training accuracy: {logistic.score()}")
    print(f"Percision score: {precision_score(y_test, y_pred)}")
    print(f"{classification_report(y_test, y_pred)}")
    
    # Most importantly - evaluate on unseen test data
    test_accuracy = logistic.model.score(X_test, y_test)
    print(f"Test accuracy: {test_accuracy}")
    print(y.value_counts(normalize=True))

if(__name__ == "__main__"):
    main()