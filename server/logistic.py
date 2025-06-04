import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score

class Logistic:

    """
        Logistic regression model for binary classification of mental health risk.
        :param data: DataFrame containing the training data
        :param target: The target variable to predict
        :return: None
    """
    def __init__(self, X, y, target: str):
        self.target = target
        self.model = LogisticRegression()

        # split the data between features and target from the dataframe
        self.X = X
        self.y = y

    def is_trained(self):
        if self.model is None: 
            return False
        elif hasattr(self.model, 'coef_') and self.model.coef_ is not None:
            return True
        return False

    def get_best_params(self):
        param_grid = {
            'C': [.0001, .001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [100, 200],
            'random_state': [42]
        }

        grid = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy')
        grid.fit(self.X, self.y)

        return grid.best_params_, grid.best_score_

    def train(self):
        if not self.is_trained():
            best_params, best_score = self.get_best_params()
            self.model = LogisticRegression(**best_params)
            self.model.fit(self.X, self.y)


        else: 
            # check if the model is already trained
            raise Exception("Model already trained!!!")

    """
        under the hood the model is using the sigmoid function to predict the probability of the target variable
        so we can just technically make a function that does the same thing but this is easier
    """
    def predict(self, new_data: pd.DataFrame):
        # check if there is a trained model
        if not self.is_trained():
            raise Exception("No Model trained yet!!!")
        
        # check if the new data has the same columns as the training data
        if not all(col in self.X.columns for col in new_data.columns):
            raise Exception("Missing columns in new data!!!")
        return self.model.predict(new_data)
    
    """
        This method returns the mean accuracy of the model on the training data
    """
    def score(self):
        # check if there is a trained model
        if not self.is_trained():
            raise Exception("No Model trained yet!!!")
        return self.model.score(self.X, self.y)
    