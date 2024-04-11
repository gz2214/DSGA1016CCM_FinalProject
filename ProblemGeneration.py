# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:22:22 2024

@author: dinos
"""

import numpy as np 
from random import sample


#%%
pH_set=[.01,.05,.1,.2,.25,.4,.5,.6,.75,.8,.9,.95,.99, 1]
tmp=[-7,-6,-5,-4,-3,-2,2,3,4,5,6,7,8]
def LotteryA():
    EVA=sample([i for i in range(-10,31)],1)[0]
    if np.random.uniform(0,1)<.4:
        LA,HA=EVA,EVA
        pHA=1
        LotNumA=1
        LotShapeA='-'
    else:
        pHA=sample(pH_set,1)[0]
        if pHA==1:
            LA,HA=EVA,EVA
        else:
            temp=np.random.triangular(-50, EVA, 120, size=None)
            if round(temp)>EVA:
                HA=round(temp)
                LA=round((EVA-HA*pHA)/(1-pHA))
            elif round(temp)<EVA:
                LA=round(temp)
                HA=round((EVA-LA*(1-pHA))/pHA)
            elif round(temp)==EVA:
                LA,HA=EVA,EVA
        p=np.random.uniform(0,1)
        if p<=.6:
            LotNumA=1
            LotShapeA='-'
        elif p>.6 and p<=.8:
            temp=sample(tmp,1)[0]
            LotNumA=abs(temp)
            if temp>0:
                LotShapeA='R-skew'
            else:
                LotShapeA='L-skew'
        else:
            LotNumA=sample([3,5,7,9],1)[0]
            LotShapeA='Symm'
    return([EVA,HA,pHA,LA,LotNumA,LotShapeA])
#%%
def LotteryB(EVA):
    notdone=True
    while notdone:
        DEV=np.random.uniform(-20,20,5).mean()
        if EVA+DEV>=-50:
            EVB=EVA+DEV
            notdone=False
    pHB=sample(pH_set,1)[0]
    if pHB==1:
        HB,LB=round(EVB),round(EVB)
    else:
        temp=np.random.triangular(-50, EVB, 120, size=None)
        if round(temp)>EVB:
            HB=round(temp)
            LB=round((EVB-HB*pHB)/(1-pHB))
        elif round(temp)<EVB:
            LB=round(temp)
            HB=round((EVB-LB*(1-pHB))/pHB)
    p=np.random.uniform(0,1)
    if p<=.5:
        LotNumB=1
        LotShapeB='-'
    elif p<.5 and p<.75:
        temp=sample(tmp,1)[0]
        LotNumB=abs(temp)
        if temp>0:
            LotShapeB='R-Skew'
        else:
            LotShapeB='L-Skew'
    else:
        LotShapeB='Symm'
        LotNumB=sample([3,5,7,9],1)[0]
        
    return([HB,pHB,LB,LotNumB,LotShapeB])
            

#%%

def ProblemGeneration():
    notDone=True
    while notDone:
        A=LotteryA()
        B=LotteryB(A[0])
        Corr=np.random.choice([-1,0,1],p=[.1,.8,.1])
        Amb=np.random.choice([0,1],p=[.8,.2])
        ###get rid of these generations
        tooBig=any(x>256 for x in [A[1],A[3],B[0],B[2]])
        tooSmall=any(x <-50 for x in [A[1],A[3],B[0],B[2]])    
        identical= (A==B and Amb==0)
        B1p=(Amb==1 and B[1]==1)
        noVar=(Corr==1 and any(x=='-' for x in [A[5],B[4]]))
        if (tooBig+tooSmall+identical+B1p+noVar)==0:
            notDone=False
    return(A[1:]+B+[Corr]+[Amb])
#%%
n=100
dataSet=[ProblemGeneration() for i in range(n)]






