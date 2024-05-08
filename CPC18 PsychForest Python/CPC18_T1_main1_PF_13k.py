import os
#os.chdir('---')
#####################################################################################
### Section A: Please change this section to import necessary files and packages ###
#####################################################################################
import pandas as pd
import numpy as np
import time
from CPC18_PF_pred import CPC18_PF_pred

from tqdm import tqdm


if __name__ == '__main__':
    ####################################################
    ### Section B: Please do not change this section ###
    ####################################################
    # load problems to predict (in this example, the estimation set problems)
    Data = pd.read_csv('c13k_imputed.csv')
    # Data = df.sample(frac=0.001, random_state=32) #comment out and replace df with Data in line above
    train_data = pd.read_csv('TrainData.csv')
    # useful variables
    nProblems = Data.shape[0]
    PredictedAll = np.zeros(shape=(nProblems, 5))
    ### End of Section A ###

    #################################################################
    ### Section C: Please change only lines 41-47 in this section ###
    #################################################################
    for prob in tqdm(range((nProblems))):
        # read problem's parameters
        Ha = Data['Ha'].iloc[prob]
        pHa = Data['pHa'].iloc[prob]
        La = Data['La'].iloc[prob]
        LotShapeA = Data['LotShapeA'].iloc[prob]
        LotNumA = Data['LotNumA'].iloc[prob]
        Hb = Data['Hb'].iloc[prob]
        pHb = Data['pHb'].iloc[prob]
        Lb = Data['Lb'].iloc[prob]
        LotShapeB = Data['LotShapeB'].iloc[prob]
        LotNumB = Data['LotNumB'].iloc[prob]
        Amb = Data['Amb'].iloc[prob]
        Corr = Data['Corr'].iloc[prob]

        # please plug in here your model that takes as input the 12 parameters
        # defined above and gives as output a vector size 5 named "Prediction"
        # in which each cell is the predicted B-rate for one block of five trials
        # example:
        Prediction = CPC18_PF_pred(train_data, Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb,
                                   Corr)
        # end of example

        PredictedAll[prob, :] = Prediction
        # for verbose progression
        print('{}: Finish problem number: {}'.format((time.asctime(time.localtime(time.time()))), prob + 1))

    ### End of Section C ###

    ####################################################
    ### Section D: Please do not change this section ###
    ####################################################
    # compute MSE
    # ObservedAll = Data[['B.1', 'B.2', 'B.3', 'B.4', 'B.5']]
    b_rate = PredictedAll.mean(axis=1)
    ObservedAll = Data['bRate']
    
    probMSEs = 100 * ((b_rate - ObservedAll) ** 2)
    totalMSE = np.mean(probMSEs)
    
    print('MSE over the {} problems: {}'.format(nProblems, totalMSE))
    # for keeping the predicted choice rates
    np.savetxt("outputAll_13k.csv", PredictedAll, delimiter=",", header="B1,B2,B3,B4,B5", fmt='%.4e')
    ### End of Section D ###
