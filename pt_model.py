import argparse
import ast
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy.special import softmax

# Utility Functions
def AsymmetricLinearUtil(outcome, lambda_al=1.0, alpha_al=1.0):
    # define positive and negative parts of the function
    util_outcome = 0
    if outcome > 0.0:
        util_outcome = alpha_al * outcome
    else:
        util_outcome = -np.abs(lambda_al) * np.abs(outcome)

    return util_outcome

def NormExpLossAverseUtil(outcome, lambda_norm=1.0, alpha_norm=1.0, beta_norm=1.0):
    util_outcome = 0
    if outcome >= 0.0:
        util_outcome = (1 / alpha_norm) * (1 - np.exp(-alpha_norm * np.abs(outcome)))
    else:
        util_outcome = (-np.abs(lambda_norm) / beta_norm) * (1 - np.exp(-beta_norm * np.abs(outcome)))

    return util_outcome

# Probability Weight Functions
def KT_PWF(p, alpha_kt=0.5):
    return (p ** alpha_kt) / ((p ** alpha_kt + (1 - p) ** alpha_kt) ** (1 / alpha_kt))

def ConstantRelativeSensitivityPWF(p, alpha_crs=0.5, beta_crs=0.5):
    p_pwf=0
    if p <= beta_crs:
        p_pwf = beta_crs ** (1 - alpha_crs) * p ** alpha_crs
    else:
        p_pwf = 1 - (1 - beta_crs) ** (1 - alpha_crs) * (1 - p) ** alpha_crs
    return p_pwf

# PT Model Class
class ProspectTheoryModel(BaseEstimator):
    def __init__(self, alpha_kt=None, alpha_al=None, lambda_al=None, lambda_norm=None, alpha_norm=None, beta_norm=None, alpha_crs=None, beta_crs=None, util_func='AsymmetricLinearUtil', pwf='KT_PWF'):
        self.alpha_kt = alpha_kt
        self.alpha_al = alpha_al
        self.lambda_al = lambda_al
        self.lambda_norm = lambda_norm
        self.alpha_norm = alpha_norm
        self.beta_norm = beta_norm
        self.alpha_crs = alpha_crs
        self.beta_crs = beta_crs
        self.util_func = util_func
        self.pwf = pwf

    def fit(self, X, y):
        # Assume fitting is trivial; store X and y for demonstrating purposes
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        if self.pwf == 'KT_PWF':
            pw = lambda p: KT_PWF(p, alpha_kt=self.alpha_kt)
        elif self.pwf == 'ConstantRelativeSensitivityPWF':
            pw = lambda p: ConstantRelativeSensitivityPWF(p, alpha_crs=self.alpha_crs, beta_crs=self.beta_crs)
        
        if self.util_func == 'AsymmetricLinearUtil':
            u = lambda outcome: AsymmetricLinearUtil(outcome, alpha_al=self.alpha_al, lambda_al=self.lambda_al)
        elif self.util_func == 'NormExpLossAverseUtil':
            u = lambda outcome: NormExpLossAverseUtil(outcome, lambda_norm=self.lambda_norm, alpha_norm=self.alpha_norm, beta_norm=self.beta_norm)
        predictions = [
            softmax([
                pw(i[0][0]) * u(i[0][1]) + pw(i[1][0]) * u(i[1][1]),
                pw(i[2][0]) * u(i[2][1]) + pw(i[3][0]) * u(i[3][1])
            ])[1] for i in X
        ]
        return predictions

    def score(self, X, y):
        # Implement scoring by negating the mean squared error
        predictions = self.predict(X)
        return -mean_squared_error(y, predictions)

    def get_params(self, deep=True):
        # This method needs to return all parameters necessary for replicating this estimator
        params = dict(
            util_func=self.util_func,
            pwf=self.pwf,
            alpha_kt = self.alpha_kt,
            alpha_al = self.alpha_al,
            lambda_al = self.lambda_al,
            lambda_norm = self.lambda_norm,
            alpha_norm = self.alpha_norm,
            beta_norm = self.beta_norm,
            alpha_crs = self.alpha_crs,
            beta_crs = self.beta_crs
        )
        return params

    def set_params(self, **params):
        # Set parameters based on input dictionary
        for key, value in params.items():
            setattr(self, key, value)
        return self
    

def ProspectTheory(param_grid, frac_data=1, util_func='AsymmetricLinearUtil', pwf='KT_PWF'):
    data=pd.read_csv("c13k_selections.csv")
    data=data.sample(frac=frac_data, random_state=42)
    
    A=[[[data['pHa'][i],data['Ha'][i]],[1-data['pHa'][i],data['La'][i]]] for i in range(14568)]
    A =np.array([i for i in A])
    B=[[[data['pHb'][i],data['Hb'][i]],[1-data['pHb'][i],data['Lb'][i]]] for i in range(14568)]
    B =np.array([i for i in B])
    X = np.concatenate((A, B), axis=1)
    Y=np.array(data['bRate'])
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = ProspectTheoryModel(util_func=util_func, pwf=pwf)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, Y_train)
    
    best_model = grid_search.best_estimator_
    Y_test_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(Y_test, Y_test_pred)

    print(f"PT with {util_func} and {pwf}")
    print("Best parameters:", grid_search.best_params_)
    print("Best score (MSE):", -grid_search.best_score_)
    print("Test MSE:", test_mse)  

    return grid_search, test_mse

if __name__ == "__main__":
    # example run: python pt_model.py --param_grid "{'alpha_kt': [0.3, 0.5, 0.7, 1, 1.2, 1.5],'alpha_al': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2],'lambda_al': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2]}"
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_grid', help='A dictionary of values for hyperparameter tuning, please input the dictionary as a string')
    parser.add_argument('--frac_data', help='fraction of the 13k data will be used', default=1)
    parser.add_argument("--util_func",help='A utility function to use', default="AsymmetricLinearUtil")
    parser.add_argument('--weight_func', help='A probability weighting function to use', default="KT_PWF")

    args = parser.parse_args()

    # Parse the string representation of the dictionary into a dictionary object
    param_grid = ast.literal_eval(args.param_grid)

    ProspectTheory(param_grid, args.frac_data, args.util_func, args.weight_func)