import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import clean_data

class MentalHealthANN:
    def __init__(self, df: pd.DataFrame = None, target: str = 'Mental_Health_Risk'):
        self.df = df
        self.target = target
        self.model = None
        self.scaler = None
        self.X = None
        self.y = None
        self.feature_importance = None
        if df is not None:
            self.preprocess_data()

    def preprocess_data(self):
        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in DataFrame.")

        self.X = self.df.drop(columns=[self.target])
        self.y = self.df[self.target]

        if self.X.isna().any().any():
            raise ValueError("NaN values remain in features after preprocessing")
        if len(self.X) != len(self.y):
            raise ValueError(f"X has {len(self.X)} samples, y has {len(self.y)} samples")

        print(f"Dataset size: {self.X.shape[0]} samples")
        print(f"Class distribution:\n{self.y.value_counts(normalize=True)}")

    def plot_correlation_heatmap(self):
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        if self.target not in self.df.columns:
            self.df[self.target] = self.y
        corr_matrix = pd.concat([self.df[numeric_cols], self.y], axis=1).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Heatmap: Mental Health and Key Features')
        plt.tight_layout()
        plt.savefig('server/heatmap.png')
        plt.close()

    def is_trained(self):
        return self.model is not None

    def train(self):
        if self.X is None or self.y is None:
            self.preprocess_data()

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )

        smote_enn = SMOTEENN(random_state=42)
        X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        ])

        pipeline.fit(X_train_res, y_train_res)
        self.model = pipeline

        y_pred = pipeline.predict(X_test)
        print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, self.X, self.y, cv=cv, scoring='f1_macro')
        print(f"\nCross-Validation F1 (Macro): {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Issue', 'Has Issue'],
                    yticklabels=['No Issue', 'Has Issue'])
        plt.title('Confusion Matrix')
        os.makedirs('server', exist_ok=True)
        plt.savefig('server/confusion_matrix.png')
        plt.close()

        perm_importance = permutation_importance(pipeline, X_train, y_train, n_repeats=10, random_state=42)
        self.feature_importance = dict(zip(self.X.columns, perm_importance.importances_mean))
        print("\nFeature Importance:")
        for feature, importance in sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")

        return self.model

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
        df = clean_data()

        # Initialize and preprocess
        ann = MentalHealthANN(df, 'Mental_Health_Risk')

        # # Generate correlation heatmap
        # ann.plot_correlation_heatmap()

        # Train model
        ann.train()
        # Test prediction
        result = ann.predict_from_dict({
            'gender': 'Female',
            'age': 20,
            'course': 'Engineering',
            'cgpa': 3.25,
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