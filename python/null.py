import pandas as pd
import numpy as np

class NullModel:
     
    def __init__(self, target_type: str = "regression"):
        self.target_type = target_type
        self.y = None
        self.pred_value = None
        self.preds = None
        
    def fit(self, y):
        self.y = y
        if self.target_type == "regression":
            self.pred_value = y.mean()
        else:
            from scipy.stats import mode
            self.pred_value = mode(y)[0][0]
            
    def get_length(self):
        return len(self.y)
    
    def predict(self, y):
        self.preds = np.full((self.get_length(), 1), self.pred_value)
        return self.preds
    
    def fit_predict(self, y):
        self.fit(y)
        return self.predict(self.y)
