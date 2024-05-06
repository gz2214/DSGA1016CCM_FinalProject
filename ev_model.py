import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.special import softmax

# EV model
def ExpectedValue(frac_data=1):
    data=pd.read_csv("c13k_selections.csv")
    data=data.sample(frac=frac_data, random_state=42)
    
    A=[[[data['pHa'][i],data['Ha'][i]],[1-data['pHa'][i],data['La'][i]]] for i in range(14568)]
    A =np.array([i for i in A])
    B=[[[data['pHb'][i],data['Hb'][i]],[1-data['pHb'][i],data['Lb'][i]]] for i in range(14568)]
    B =np.array([i for i in B])
    X = np.concatenate((A, B), axis=1)
    Y=np.array(data['bRate'])
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Since it's EV, we can directly use test data to evaluate
    Y_test_pred = [softmax([i[0][0]*i[0][1]+i[1][0]*i[1][1], i[2][0]*i[2][1]+i[3][0]*i[3][1]])[1] for i in X_test]
        
    ev_mse = mean_squared_error(Y_test, Y_test_pred)
    print(f"The mse of Expected Value is {ev_mse}")
    return ev_mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frac_data', help='fraction of the 13k data will be used', default=1)
    args = parser.parse_args()
    ExpectedValue(args.frac_data)
    
