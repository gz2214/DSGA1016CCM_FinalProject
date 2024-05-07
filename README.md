# DSGA1016CCM_FinalProject


## Current Baseline NN Results for 13k

| Epoch | Training Loss | Validation Loss | Time/Step | Total Time |
|-------|---------------|-----------------|-----------|------------|
|   1   |     2.7335    |      0.0801     |    1s     |   365ms    |
|   2   |     0.0581    |      0.0508     |    0s     |   365ms    |
|   3   |     0.0526    |      0.0569     |    0s     |   365ms    |
|   4   |     0.0540    |      0.0520     |    0s     |   365ms    |
|   5   |     0.0542    |      0.0556     |    0s     |   365ms    |
|   6   |     0.0574    |      0.0499     |    0s     |   365ms    |
|   7   |     0.0531    |      0.0507     |    0s     |   365ms    |
|   8   |     0.0534    |      0.0504     |    0s     |   365ms    |
|   9   |     0.0536    |      0.0492     |    0s     |   365ms    |
|  10   |     0.0519    |      0.0494     |    0s     |   365ms    |


## Current Baseline NN Results for CPC 18

| Epoch | Training Loss | Validation Loss | Time/Step | Total Time |
|-------|---------------|-----------------|-----------|------------|
|   1   |     0.1756    |      0.0893     |    1s     |   75ms     |
|   2   |     0.1752    |      0.0891     |    0s     |   16ms     |
|   3   |     0.1752    |      0.0893     |    0s     |   22ms     |
|   4   |     0.1750    |      0.0891     |    0s     |   13ms     |
|   5   |     0.1746    |      0.0886     |    0s     |   16ms     |
|   6   |     0.1745    |      0.0886     |    0s     |   15ms     |
|   7   |     0.1744    |      0.0889     |    0s     |   15ms     |
|   8   |     0.1743    |      0.0890     |    0s     |   15ms     |
|   9   |     0.1744    |      0.0891     |    0s     |   15ms     |
|  10   |     0.1743    |      0.0891     |    0s     |   14ms     |

## Expected Value
Validation MSE: 0.2235

## Prospect Theory
|   Utility Function  | Probability Weight Function  | Training Loss | Validation Loss |
|---------------------|------------------------------|---------------|-----------------|
| AsymmetricLinearUtil|            KT_PWF            |     0.0495    |     0.0507      |
|NormExpLossAverseUtil|            KT_PWF            |     0.0493    |     0.0505      |
| AsymmetricLinearUtil|ConstantRelativeSensitivityPWF|     0.0492    |     0.0505      |
|NormExpLossAverseUtil|ConstantRelativeSensitivityPWF|     0.0492    |     0.0506      |

## Prospect Theory with NN as the utility and probability weight functions for 13k

| Epoch | Training Loss | Validation Loss | Time/Step | Total Time |
|-------|---------------|-----------------|-----------|------------|
|   1   |     0.0511    |      0.0506     |    5s     |   365ms    |
|   2   |     0.0487    |      0.0508     |    4s     |   365ms    |
|   3   |     0.0490    |      0.0506     |    4s     |   365ms    |
|   4   |     0.0494    |      0.0504     |    5s     |   365ms    |
|   5   |     0.0495    |      0.0507     |    4s     |   365ms    |
|   6   |     0.0493    |      0.0505     |    4s     |   365ms    |
|   7   |     0.0492    |      0.0508     |    4s     |   365ms    |
|   8   |     0.0486    |      0.0504     |    4s     |   365ms    |
|   9   |     0.0497    |      0.0505     |    4s     |   365ms    |
|  10   |     0.0486    |      0.0505     |    4s     |   365ms    |



