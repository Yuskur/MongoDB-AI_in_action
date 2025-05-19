import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
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
        required_cols = ['Choose your gender', 'What is your course?', 'Your current year of Study',
                        'What is your CGPA?', 'Do you have Depression?', 'Do you have Anxiety?',
                        'Do you have Panic attack?', 'Age']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")

        # Create target variable
        try:
            self.df[self.target] = (
                (self.df['Do you have Depression?'] == 'Yes').astype(int) |
                (self.df['Do you have Anxiety?'] == 'Yes').astype(int) |
                (self.df['Do you have Panic attack?'] == 'Yes').astype(int)
            )
        except KeyError as e:
            raise ValueError(f"Error creating target variable: {str(e)}")

        # Handle missing Age
        self.df['Age'] = pd.to_numeric(self.df['Age'], errors='coerce').fillna(self.df['Age'].median())

        # Encode categorical features
        try:
            self.label_encoders['gender'] = LabelEncoder()
            self.df['Gender'] = self.label_encoders['gender'].fit_transform(
                self.df['Choose your gender'].str.lower().fillna('unknown'))
        except Exception as e:
            raise ValueError(f"Error encoding Gender: {str(e)}")

        # Group majors to reduce sparsity
        stem_majors = ['engineering', 'bit', 'bcs', 'it', 'biomedical science', 'marine science',
                       'biotechnology', 'radiography', 'nursing', 'cts', 'koe', 'engin', 'engine']
        social_majors = ['psychology', 'human sciences', 'laws', 'law', 'communication', 'islamic education',
                         'pendidikan islam', 'usuluddin', 'fiqh', 'fiqh fatwa', 'human resources',
                         'business administration', 'banking studies', 'econs', 'diploma tesl']
        try:
            self.df['Major_Group'] = self.df['What is your course?'].str.lower().apply(
                lambda x: 'STEM' if isinstance(x, str) and any(m in x.lower() for m in stem_majors) else
                          'Social Sciences' if isinstance(x, str) and any(m in x.lower() for m in social_majors) else 'Other')
            self.label_encoders['major'] = LabelEncoder()
            self.df['Major'] = self.label_encoders['major'].fit_transform(self.df['Major_Group'])
        except Exception as e:
            raise ValueError(f"Error encoding Major: {str(e)}")

        # Extract year number and convert to float
        try:
            self.df['Year'] = self.df['Your current year of Study'].str.lower().str.extract(r'(\d+)').astype(float).fillna(1.0)
        except Exception as e:
            raise ValueError(f"Error processing Year: {str(e)}")

        # Convert GPA ranges to numerical values
        gpa_map = {
            '0 - 1.99': 1.0, '2.00 - 2.49': 2.25, '2.50 - 2.99': 2.75,
            '3.00 - 3.49': 3.25, '3.50 - 4.00': 3.75
        }
        try:
            self.df['GPA'] = self.df['What is your CGPA?'].map(gpa_map).fillna(3.0)
        except Exception as e:
            raise ValueError(f"Error processing GPA: {str(e)}")

        # Select features and handle missing values
        features = ['Gender', 'Age', 'Major', 'Year', 'GPA']
        try:
            self.df[features] = self.df[features].fillna(self.df[features].mean())
            self.X = self.df[features]
            self.y = self.df[self.target]
        except KeyError as e:
            raise ValueError(f"Error selecting features: {str(e)}")

        print(f"Dataset size: {self.X.shape[0]} samples")
        print(f"Class distribution:\n{self.y.value_counts()}")

    def plot_correlation_heatmap(self):
        if self.X is None or self.y is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        corr_matrix = self.df[['Gender', 'Age', 'Major', 'Year', 'GPA', self.target]].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Heatmap: Mental Health and Features')
        plt.tight_layout()
        plt.savefig('heatmap.png')
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
        smote = SMOTE(random_state=42, k_neighbors=3)  # Reduced k_neighbors for small dataset
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(10, 5),  # Reduced to prevent overfitting
                activation='relu',
                solver='adam',
                alpha=0.01,  # Stronger regularization
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
        print(classification_report(y_test, y_pred))

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
        plt.savefig('confusion_matrix.png')
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
            # Prepare input DataFrame with column names
            major = str(input_data.get('major', 'STEM')).lower()
            if major not in self.label_encoders['major'].classes_:
                major = 'STEM'  # Fallback to a known category
            input_df = pd.DataFrame([{
                'Gender': self.label_encoders['gender'].transform(
                    [str(input_data.get('gender', 'Male')).lower()])[0],
                'Age': float(input_data.get('age', 20)),
                'Major': self.label_encoders['major'].transform([major])[0],
                'Year': float(input_data.get('year', 2)),
                'GPA': float(input_data.get('gpa', 3.5))
            }], columns=['Gender', 'Age', 'Major', 'Year', 'GPA'])

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
            raise Exception("No Model trained yet!")
        if not all(col in self.X.columns for col in new_data.columns):
            raise Exception("Missing columns in new data!")
        return self.model.predict(new_data)

    def predict_proba(self, new_data: pd.DataFrame):
        if not self.is_trained():
            raise Exception("No Model trained yet!")
        if not all(col in self.X.columns for col in new_data.columns):
            raise Exception("Missing columns in new data!")
        return self.model.predict_proba(new_data)[:, 1]

    def score(self):
        if not self.is_trained():
            raise Exception("No Model trained yet!")
        return self.model.score(self.X, self.y)

if __name__ == "__main__":
    try:
        # Load data
        csv_path = os.path.join(os.path.dirname(__file__), 'student-mental-health.csv')
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
            'major': 'Engineering',
            'year': 2,
            'gpa': 3.25
        })

        if result:
            print(f"\nPrediction: {'At Risk' if result['prediction'] else 'Not At Risk'}")
            print(f"Probability: {result['probability']:.2f}")
            print(f"Risk Level: {result['risk']}")
            print(f"Model Accuracy: {ann.score():.2f}")

    except Exception as e:
        print(f"Error: {str(e)}")