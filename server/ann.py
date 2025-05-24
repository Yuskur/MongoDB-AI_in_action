import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class MentalHealthANN:
    def __init__(self, df: pd.DataFrame = None, target: str = 'Mental_Health_Issue'):
        self.df = df
        self.target = target
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.X = None
        self.y = None
        self.feature_importance = None
        if df is not None:
            self.preprocess_data()

    def preprocess_data(self):
        # Validate required columns
        required_cols = [
            'Age', 'Course', 'Gender', 'CGPA', 'Stress_Level', 'Depression_Score',
            'Anxiety_Score', 'Sleep_Quality', 'Physical_Activity', 'Diet_Quality',
            'Social_Support', 'Relationship_Status', 'Substance_Use',
            'Counseling_Service_Use', 'Family_History', 'Chronic_Illness',
            'Financial_Stress', 'Extracurricular_Involvement', 'Semester_Credit_Load',
            'Residence_Type'
        ]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")

        # Create target variable
        try:
            self.df[self.target] = (
                (self.df['Depression_Score'] >= 3).astype(int) |
                (self.df['Anxiety_Score'] >= 3).astype(int) |
                (self.df['Stress_Level'] >= 3).astype(int)
            )
        except KeyError as e:
            raise ValueError(f"Error creating target variable: {str(e)}")

        # Initialize processed DataFrame
        processed_data = pd.DataFrame(index=self.df.index)

        # Handle numerical features
        numerical_cols = [
            'Age', 'CGPA', 'Stress_Level', 'Depression_Score', 'Anxiety_Score',
            'Semester_Credit_Load'
        ]
        imputer = SimpleImputer(strategy='median')
        numerical_data = pd.DataFrame(
            imputer.fit_transform(self.df[numerical_cols]),
            columns=numerical_cols,
            index=self.df.index
        )
        processed_data = processed_data.join(numerical_data)

        # Handle categorical features with ordinal encoding
        categorical_cols = [
            'Gender', 'Course', 'Sleep_Quality', 'Physical_Activity', 'Diet_Quality',
            'Social_Support', 'Relationship_Status', 'Substance_Use',
            'Counseling_Service_Use', 'Family_History', 'Chronic_Illness',
            'Residence_Type', 'Financial_Stress', 'Extracurricular_Involvement'
        ]
        ordinal_mappings = {
            'Financial_Stress': {'unknown': 0, 'low': 0, 'moderate': 1, 'high': 2},
            'Extracurricular_Involvement': {'unknown': 0, 'low': 0, 'moderate': 1, 'high': 2},
            'Sleep_Quality': {'unknown': 0, 'poor': 0, 'average': 1, 'good': 2},
            'Physical_Activity': {'unknown': 0, 'low': 0, 'moderate': 1, 'high': 2},
            'Diet_Quality': {'unknown': 0, 'poor': 0, 'average': 1, 'good': 2},
            'Social_Support': {'unknown': 0, 'low': 0, 'moderate': 1, 'high': 2}
        }
        for col in categorical_cols:
            try:
                self.df[col] = self.df[col].replace(['', ' ', None, np.nan], 'unknown').astype(str).str.lower()
                if col in ordinal_mappings:
                    processed_data[col] = self.df[col].map(ordinal_mappings[col]).fillna(0).astype(int)
                else:
                    self.label_encoders[col] = LabelEncoder()
                    processed_data[col] = self.label_encoders[col].fit_transform(self.df[col])
            except Exception as e:
                raise ValueError(f"Error encoding {col}: {str(e)}")

        # Validate column lengths
        for col in processed_data.columns:
            if len(processed_data[col]) != len(self.df):
                raise ValueError(f"Column {col} has {len(processed_data[col])} samples, expected {len(self.df)}")

        # Select features and ensure no NaNs
        features = numerical_cols + categorical_cols
        try:
            self.X = processed_data[features]
            self.y = self.df[self.target]
            if self.X.isna().any().any():
                raise ValueError("NaN values remain in features after preprocessing")
            if len(self.X) != len(self.y):
                raise ValueError(f"X has {len(self.X)} samples, y has {len(self.y)} samples")
        except KeyError as e:
            raise ValueError(f"Error selecting features: {str(e)}")

        print(f"Dataset size: {self.X.shape[0]} samples")
        print(f"Class distribution:\n{self.y.value_counts(normalize=True)}")

    def plot_correlation_heatmap(self):
        if self.X is None or self.y is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        corr_cols = ['Age', 'CGPA', 'Stress_Level', 'Depression_Score', 'Anxiety_Score', self.target]
        corr_matrix = pd.concat([self.X[corr_cols[:-1]], self.y], axis=1).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Heatmap: Mental Health and Key Features')
        plt.tight_layout()
        plt.savefig('server/heatmap.png')
        plt.close()

    def is_trained(self):
        if self.model is None:
            return False
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(self.model)
            return True
        except:
            return False

    def train(self, max_iter=3000):
        if self.X is None or self.y is None:
            self.preprocess_data()

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y)

        # Handle class imbalance with SMOTE
        minority_count = min(y_train.value_counts())
        k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(20, 10),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=max_iter,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2
            ))
        ])

        pipeline.fit(X_train_res, y_train_res)
        self.model = pipeline
        self.scaler = pipeline.named_steps['scaler']

        # Evaluation
        y_pred = pipeline.predict(X_test)
        print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Cross-validation
        cv_scores = cross_val_score(pipeline, self.X, self.y, cv=5, scoring='recall_macro')
        print(f"\nCross-Validation Recall (Macro): {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Issue', 'Has Issue'],
                    yticklabels=['No Issue', 'Has Issue'])
        plt.title('Confusion Matrix')
        plt.savefig('server/confusion_matrix.png')
        plt.close()

        # Feature importance
        perm_importance = permutation_importance(pipeline, X_train, y_train, n_repeats=10, random_state=42)
        self.feature_importance = dict(zip(self.X.columns, perm_importance.importances_mean))
        print("\nFeature Importance:")
        for feature, importance in sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")

        return self.model

    def predict_from_dict(self, input_data: dict):
        if not self.is_trained():
            raise Exception("Model not trained. Call train() first.")

        try:
            input_dict = {}
            for col in self.X.columns:
                if col in self.label_encoders:
                    val = str(input_data.get(col.lower(), 'unknown')).lower()
                    if val not in self.label_encoders[col].classes_:
                        val = 'unknown'
                    input_dict[col] = self.label_encoders[col].transform([val])[0]
                elif col in ['Financial_Stress', 'Extracurricular_Involvement', 'Sleep_Quality',
                            'Physical_Activity', 'Diet_Quality', 'Social_Support']:
                    mapping = {
                        'Financial_Stress': {'unknown': 0, 'low': 0, 'moderate': 1, 'high': 2},
                        'Extracurricular_Involvement': {'unknown': 0, 'low': 0, 'moderate': 1, 'high': 2},
                        'Sleep_Quality': {'unknown': 0, 'poor': 0, 'average': 1, 'good': 2},
                        'Physical_Activity': {'unknown': 0, 'low': 0, 'moderate': 1, 'high': 2},
                        'Diet_Quality': {'unknown': 0, 'poor': 0, 'average': 1, 'good': 2},
                        'Social_Support': {'unknown': 0, 'low': 0, 'moderate': 1, 'high': 2}
                    }[col]
                    val = str(input_data.get(col.lower(), 'unknown')).lower()
                    input_dict[col] = mapping.get(val, 0)
                else:
                    input_dict[col] = float(input_data.get(col.lower(), self.X[col].median()))

            input_df = pd.DataFrame([input_dict], columns=self.X.columns)
            proba = self.model.predict_proba(input_df)[0]
            prediction = int(self.model.predict(input_df)[0])

            return {
                'prediction': prediction,
                'probability': float(proba[1]),
                'risk': 'High' if proba[1] > 0.7 else 'Medium' if proba[1] > 0.4 else 'Low'
            }
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None

    def predict(self, new_data: pd.DataFrame):
        if not self.is_trained():
            raise Exception("No Model trained yet")
        if not all(col in self.X.columns for col in new_data.columns):
            raise Exception("Missing columns in new data")
        return self.model.predict(new_data)

    def predict_proba(self, new_data: pd.DataFrame):
        if not self.is_trained():
            raise Exception("No Model trained yet")
        if not all(col in self.X.columns for col in new_data.columns):
            raise Exception("Missing columns in new data")
        return self.model.predict_proba(new_data)[:, 1]

    def score(self):
        if not self.is_trained():
            raise Exception("No Model trained yet")
        return self.model.score(self.X, self.y)

if __name__ == "__main__":
    try:
        # Load data
        csv_path = os.path.join(os.path.dirname(__file__), 'students_mental_health_survey.csv')
        df = pd.read_csv(csv_path)

        # Initialize and preprocess
        ann = MentalHealthANN(df, 'Mental_Health_Issue')

        # Generate correlation heatmap
        ann.plot_correlation_heatmap()

        # Train model
        ann.train(max_iter=3000)

        # Test prediction
        result = ann.predict_from_dict({
            'gender': 'Female',
            'age': 20,
            'course': 'Engineering',
            'cgpa': 3.25,
            'stress_level': 2,
            'depression_score': 1,
            'anxiety_score': 1,
            'sleep_quality': 'Good',
            'physical_activity': 'Moderate',
            'diet_quality': 'Average',
            'social_support': 'Moderate',
            'relationship_status': 'Single',
            'substance_use': 'Never',
            'counseling_service_use': 'Never',
            'family_history': 'No',
            'chronic_illness': 'No',
            'financial_stress': 'Moderate',
            'extracurricular_involvement': 'Moderate',
            'semester_credit_load': 20,
            'residence_type': 'Off-Campus'
        })

        if result:
            print(f"\nPrediction: {'At Risk' if result['prediction'] else 'Not At Risk'}")
            print(f"Probability: {result['probability']:.2f}")
            print(f"Risk Level: {result['risk']}")
            print(f"Model Accuracy: {ann.score():.2f}")

    except Exception as e:
        print(f"Error: {str(e)}")