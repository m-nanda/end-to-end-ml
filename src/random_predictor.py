import numpy as np
from sklearn.base import BaseEstimator

class RandomPredictor(BaseEstimator):
    def __init__(self, classes) -> None:
        """
        Initialize the RandomPredictor with a set of possible classes.
        
        Parameters:
        classes (list or array-like): A list of possible class labels.
        """
        self.classes = classes

    def predict(self, X):
        """
        Generate random predictions for each input sample.
        
        Parameters:
        X (array-like): The input data. The size of the input determines how many predictions to make.
        
        Returns:
        array-like: Random predictions from the specified classes.
        """
        # Generate random predictions for each sample in X
        random_predictions = np.random.choice(self.classes, size=len(X))
        return random_predictions

    def predict_proba(self, X):
        """
        Generate random probability predictions for each input sample.
        
        Parameters:
        X (array-like): The input data. The size of the input determines how many predictions to make.
        
        Returns:
        array-like: Random predictions from the specified classes.
        """
        # Generate random probability predictions for each sample in X
        proba_predictions = np.random.rand(len(X), len(self.classes))
        return proba_predictions
    
    def get_params(self, deep=True):
        # Return parameters as a dictionary (important for grid search)
        return {'n_classes': self.n_classes}
    
    def set_params(self, **params):
        # Set parameters from a dictionary (important for grid search)
        for key, value in params.items():
            setattr(self, key, value)
        return self