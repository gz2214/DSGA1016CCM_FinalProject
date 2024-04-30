import argparse
import json

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
import pandas as pd 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from scipy import stats

import torch

from tensorflow.keras.models import Model

def CPC18_getDist(gamble):
    # Extract true full distributions of an option in CPC18
    #   input is high outcome (H: int), its probability (pH: double), low outcome
    #   (L: int), the shape of the lottery ('-'/'Symm'/'L-skew'/'R-skew' only), and
    #   the number of outcomes in the lottery (lot_num: int)
    #   output is a matrix (numpy matrix) with first column a list of outcomes (sorted
    #   ascending) and the second column their respective probabilities.
    H = gamble[:, 0]
    pH = gamble[:, 1]
    L = gamble[:, 2]
    lot_shape = gamble[:, 3]
    lot_num = gamble[:, 4]
    
    if lot_shape == '-':
        if pH == 1:
            dist = np.array([H, pH])
            dist.shape = (1, 2)
        else:
            dist = np.array([[L, 1-pH], [H, pH]])

    else:  # H is multi outcome
        # compute H distribution
        high_dist = np.zeros(shape=(lot_num, 2))
        if lot_shape == 'Symm':
            k = lot_num - 1
            for i in range(0, lot_num):
                high_dist[i, 0] = H - k / 2 + i
                high_dist[i, 1] = pH * stats.binom.pmf(i, k, 0.5)

        elif (lot_shape == 'R-skew') or (lot_shape == 'L-skew'):
            if lot_shape == 'R-skew':
                c = -1 - lot_num
                dist_sign = 1
            else:
                c = 1 + lot_num
                dist_sign = -1
            for i in range(1, lot_num+1):
                high_dist[i - 1, 0] = H + c + dist_sign * pow(2, i)
                high_dist[i - 1, 1] = pH / pow(2, i)

            high_dist[lot_num - 1, 1] = high_dist[lot_num - 1, 1] * 2

        # incorporate L into the distribution
        dist = np.copy(high_dist)
        locb = np.where(high_dist[:, 0] == float(L))
        if locb[0].size > 0:
            dist[locb, 1] += (1-pH)
        elif pH < 1:
            dist = np.vstack((dist, [L, 1-pH]))

        dist = dist[np.argsort(dist[:, 0])]

    return dist


def get_PF_Features(df):
    # Finds the values of the engineered features that are part of Psychological Forest
    # Gets as input the parameters defining the choice problem in CPC18 and returns
    # as output a pandas data frame with this problem's features
    gamble_A = df[:, 0:5]
    gamble_B = df[:, 5:10]
    
    # Compute "naive" and "psychological" features as per Plonsky, Erev, Hazan, and Tennenholtz, 2017
    DistA = CPC18_getDist(gamble_A)
    DistB = CPC18_getDist(gamble_B)
    diffEV = (np.matrix.dot(DistB[:, 0], DistB[:, 1]) - np.matrix.dot(DistA[:, 0], DistA[:, 1]))
    diffSDs = (getSD(DistB[:, 0], DistB[:, 1]) - getSD(DistA[:, 0], DistA[:, 1]))
    MinA = DistA[0, 0]
    MinB = DistB[0, 0]
    diffMins = MinB - MinA
    nA = DistA.shape[0]  # num outcomes in A
    nB = DistB.shape[0]  # num outcomes in B
    MaxA = DistA[nA - 1, 0]
    MaxB = DistB[nB - 1, 0]
    diffMaxs = MaxB - MaxA

    diffUV = (np.matrix.dot(DistB[:, 0], np.repeat([1 / nB], nB))) - (np.matrix.dot(DistA[:, 0], np.repeat([1 / nA], nA)))
    if Amb == 1:
        ambiguous = True
    else:
        ambiguous = False

    MaxOutcome = max(MaxA, MaxB)
    SignMax = np.sign(MaxOutcome)
    if MinA == MinB:
        RatioMin = 1
    elif np.sign(MinA) == np.sign(MinB):
        RatioMin = min(abs(MinA), abs(MinB)) / max(abs(MinA), abs(MinB))
    else:
        RatioMin = 0

    Range = MaxOutcome - min(MinA, MinB)
    diffSignEV = (Range * np.matrix.dot(np.sign(DistB[:, 0]), DistB[:, 1]) -
                  Range * np.matrix.dot(np.sign(DistA[:, 0]), DistA[:, 1]))
    trivial = CPC15_isStochasticDom(DistA, DistB)
    whchdom = trivial['which'][0]
    Dom = 0
    if trivial['dom'][0] and whchdom == 'A':
        Dom = -1
    if trivial['dom'][0] and whchdom == 'B':
        Dom = 1
    BEVa = np.matrix.dot(DistA[:, 0], DistA[:, 1])
    if ambiguous:
        UEVb = np.matrix.dot(DistB[:, 0], np.repeat(1 / nB, nB))
        BEVb = (UEVb + BEVa + MinB) / 3
        pEstB = np.repeat([float(nB)], 1)  # estimation of probabilties in Amb
        t_SPminb = (BEVb - np.mean(DistB[1:nB + 1, 0])) / (MinB - np.mean(DistB[1:nB + 1, 0]))
        if t_SPminb < 0:
            pEstB[0] = 0
        elif t_SPminb > 1:
            pEstB[0] = 1
        else:
            pEstB[0] = t_SPminb
        pEstB = np.append(pEstB, np.repeat([(1 - pEstB[0]) / (nB - 1)], nB - 1))
    else:
        pEstB = DistB[:, 1]
        BEVb = np.matrix.dot(DistB[:, 0], pEstB)

    diffBEV0 = (BEVb - BEVa)
    BEVfb = (BEVb + (np.matrix.dot(DistB[:, 0], DistB[:, 1]))) / 2
    diffBEVfb = (BEVfb - BEVa)

    sampleDistB = np.column_stack((DistB[:, 0], pEstB))
    probsBetter = get_pBetter(DistA, sampleDistB, corr=1)
    pAbetter = probsBetter[0]
    pBbetter = probsBetter[1]
    pBbet_Unbiased1 = pBbetter - pAbetter

    sampleUniDistA = np.column_stack((DistA[:, 0], np.repeat([1 / nA], nA)))
    sampleUniDistB = np.column_stack((DistB[:, 0], np.repeat([1 / nB], nB)))
    probsBetterUni = get_pBetter(sampleUniDistA, sampleUniDistB, corr=1)
    pBbet_Uniform = probsBetterUni[1] - probsBetterUni[0]

    sampleSignA = np.copy(DistA)
    sampleSignA[:, 0] = np.sign(sampleSignA[:, 0])
    sampleSignB = np.column_stack((np.sign(DistB[:, 0]), pEstB))
    probsBetterSign = get_pBetter(sampleSignA, sampleSignB, corr=1)
    pBbet_Sign1 = probsBetterSign[1] - probsBetterSign[0]
    sampleSignBFB = np.column_stack((np.sign(DistB[:, 0]), DistB[:, 1]))
    if Corr == 1:
        probsBetter = get_pBetter(DistA, DistB, corr=1)
        probsBetterSign = get_pBetter(sampleSignA, sampleSignBFB, corr=1)
    elif Corr == -1:
        probsBetter = get_pBetter(DistA, DistB, corr=-1)
        probsBetterSign = get_pBetter(sampleSignA, sampleSignBFB, corr=-1)
    else:
        probsBetter = get_pBetter(DistA, DistB, corr=0)
        probsBetterSign = get_pBetter(sampleSignA, sampleSignBFB, corr=0)

    pBbet_UnbiasedFB = probsBetter[1] - probsBetter[0]
    pBbet_SignFB = probsBetterSign[1] - probsBetterSign[0]

    # convert lot shape: '-'/'Symm'/'L-skew'/'R-skew' to 4 different features for the RF model
    lot_shape_listA = lot_shape_convert(LotShapeA)
    lot_shape_listB = lot_shape_convert(LotShapeB)

    # create features data frame
    feats_labels = ('Ha', 'pHa', 'La', 'lot_shape__A', 'lot_shape_symm_A', 'lot_shape_L_A', 'lot_shape_R_A', 'LotNumA',
                    'Hb', 'pHb', 'Lb', 'lot_shape__B', 'lot_shape_symm_B', 'lot_shape_L_B', 'lot_shape_R_B', 'LotNumB',
                    'Amb', 'Corr', 'diffEV', 'diffSDs', 'diffMins', 'diffMaxs', 'diffUV', 'RatioMin', 'SignMax',
                    'pBbet_Unbiased1', 'pBbet_UnbiasedFB', 'pBbet_Uniform', 'pBbet_Sign1', 'pBbet_SignFB', 'Dom',
                    'diffBEV0', 'diffBEVfb', 'diffSignEV')
    data_lists = [[Ha, pHa, La], lot_shape_listA, [LotNumA, Hb, pHb, Lb], lot_shape_listB, [LotNumB, Amb, Corr,
                             diffEV, diffSDs, diffMins, diffMaxs, diffUV, RatioMin, SignMax, pBbet_Unbiased1,
                             pBbet_UnbiasedFB, pBbet_Uniform, pBbet_Sign1, pBbet_SignFB, Dom, diffBEV0,
                             diffBEVfb, diffSignEV]]
    features_data = [item for sublist in data_lists for item in sublist]
    tmpFeats = pd.DataFrame(features_data, index=feats_labels).T

    # duplicate features data frame as per number of blocks
    Feats = pd.concat([tmpFeats] * 5)

    # get BEAST model prediction as feature
    # beastPs = CPC15_BEASTpred(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr)
    # Feats['BEASTpred'] = beastPs

    Feats['block'] = np.arange(1, 6)
    Feats['Feedback'] = 1
    Feats.loc[Feats['block'] == 1, 'Feedback'] = 0

    return Feats


# To compute the distribution's standard deviation
def getSD(vals, probs):
    m = np.matrix.dot(vals, probs.T)
    sqds = np.power((vals - m), 2)
    var = np.matrix.dot(probs, sqds.T)
    return math.sqrt(var)


# Convert lot shape feautre to vector of 4 features
def lot_shape_convert(lot_shape):
    return {
        '-': [1, 0, 0, 0],
        'Symm': [0, 1, 0, 0],
        'L-skew': [0, 0, 1, 0],
        'R-skew': [0, 0, 0, 1],
    }[lot_shape]

def binary_to_integer(binary_list):
    return int(''.join(map(str, binary_list)), 2)

def read(input: str):
    """
    :param input: 13k or cpc18_estset.csv
    :param method: 13k<>b_rate (one output), cpc18<>b1--b5 (5 outputs)
    """
    if input == '13k':
        data = pd.read_csv("c13k_selections.csv")
        A=[[[data['pHa'][i],data['Ha'][i]],[1-data['pHa'][i],data['La'][i]]] for i in range(len(data))]
        A =np.array([i for i in A])
        print(A.shape)
        # print(A)

        B=[[[data['pHb'][i],data['Hb'][i]],[1-data['pHb'][i],data['Lb'][i]]] for i in range(len(data))]
        B =np.array([i for i in B])

        X = np.concatenate((A, B), axis=1)
        Y=np.array(data['bRate'])
        model(input, X, Y)
        
    elif input == 'CPC18':
        data = pd.read_csv("CPC18_EstSet.csv")

        data['LotShapeA'] = data['LotShapeA'].map(lot_shape_convert)
        data['LotShapeB'] = data['LotShapeB'].map(lot_shape_convert)
        data['LotShapeA'] = data['LotShapeA'].apply(binary_to_integer)
        data['LotShapeB'] = data['LotShapeB'].apply(binary_to_integer)
        
        X = np.array(data.iloc[:, 1:13])
        Y = np.array(data.iloc[:, 13:])
        
        model(input, X, Y)
    
def model(method: str, X, Y):
    
    if method=='13k':
        model = Sequential([
            Dense(64, activation='relu', input_shape=(4, 2)),  # Input shape is (4, 2) for each data point
            Dense(64, activation='relu'),
            Dense(1)  # Output layer without activation function for regression task
        ])
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        learning_rate = 0.01  # Custom learning rate
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

        # Train the model
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=32, verbose=1)
    
    
    elif method=='CPC18':
        model_cpc = Sequential([
            Embedding(input_dim=2, output_dim=4, name='lotshape_embedding'),
            Flatten(),
            Dense(64, activation='relu', input_shape=(12, )), ## 12 values as input, output is 5 predicted values for 
            Dropout(0.2),  # Add dropout for regularization
            Dense(64, activation='relu'),
            Dense(5, activation='softmax')
        ])
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        learning_rate = 0.01  # Custom learning rate
        model_cpc.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy')

        # Train the model
        model_cpc.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=32, verbose=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Choose the dataset you want to use', default=None)
    # parser.add_argument("--method", required=True)

    args = parser.parse_args()

    read(args.data)