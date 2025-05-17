import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class Logistic:
    df = None
    target = None
    model = None
    X = None
    y = None

    """
        Logistic regression model for binary classification of mental health risk.
        :param data: DataFrame containing the training data
        :param target: The target variable to predict
        :return: None
    """
    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target
        self.model = LogisticRegression()

        # split the data between features and target from the dataframe
        self.X = self.df.drop(columns=[self.target]) # input features columns
        self.y = self.df[self.target] # target variable column

    def is_trained(self):
        if self.model is None: 
            return False
        elif hasattr(self.model, 'coef_') and self.model.coef_ is not None:
            return True
        return False

    def train(self):
        self.model.fit(self.X, self.y)
        return self.model
    
    """
        under the hood the model is using the sigmoid function to predict the probability of the target variable
        so we can just technically make a function that does the same thing but this is easier
    """
    def predict(self, new_data: pd.DataFrame):
        # check if there is a trained model
        if self.model is None:
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
        if self.model is None:
            raise Exception("No Model trained yet!!!")
        
        return self.model.score(self.X, self.y)