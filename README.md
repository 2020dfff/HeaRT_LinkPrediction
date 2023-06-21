# HeaRT

Official code for the paper ["Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking"](https://arxiv.org/pdf/2306.10453.pdf).


## Installation

The experiments were run using python 3.9. The required packages and their versions can be installed via the `requirements.txt` file. 
```
pip install -r requirements.txt
``` 

One exception is the original PEG code (`benchmarking/baseline_models/PEG`), which were run using python 3.7 and the packages in the `peg_requirements.txt` file.


## Download Data

All data can be downloaded via the following:
```
cd HeaRT  # Must be in the root directory
curl https://cse.msu.edu/~shomerha/HeaRT-Data/dataset.tar.gz | tar -xvz
``` 
This includes the negative samples generated by HeaRT and the splits for Cora, Citeseer, and Pubmed. The data for the OGB datasets will be automatically downloaded from the `ogb` package.

Please note that the resulting directory `dataset` must be placed in the root project directory.


## Reproduce Results

To reproduce the results, please refer to the settings in the **scripts/hyparameters** directory. We include a file for each dataset which includes the command to train and evaluate each possible method.

For example, to reproduce the results on ogbl-collab under the existing evaluation setting, the command for each method can be found in the `ogbl-collab.sh` file located in the `scripts/hyperparameter/existing_setting_ogb/` directory.


## Generate Negative Samples using HeaRT

The set of negative samples generated by HeaRT, that were used in the study, can be reproduced via the scripts in the `scripts/HeaRT/` directory. 

A custom set of negative samples can be produced by running the `heart_negatives/create_heart_negatives.py` script. Multiple options exist to customize the negative samples. This includes:
- The CN metric used. Can be either `CN` or `RA` (default is `RA`). Specified via the `--cn-metric` argument.
- The aggregation function used. Can be either `min` or `mean` (default is `min`). Specified via the `--agg` argument.
- The number of negatives generated per positive sample. Specified via the `--num-samples` argument (default is 500).
- The PPR parameters. This includes the tolerance used for approximating the PPR (`--eps` argument) and the teleporation probability (`--alpha` argument). `alpha` is fixed at 0.15 for all datasets. For the tolerance, `eps`, we recommend following the settings found in `scripts/HeaRT`.


## Cite


