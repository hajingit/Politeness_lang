# works of language team

## 1. Modified Politeness API for backend

## 2. Find Feature

#### package used:
[`NumPy`](http://www.numpy.org/)  
[`Pandas`](https://pandas.pydata.org/)  
[`SciPy`](http://www.scipy.org/)  
[`seaborn`](https://seaborn.pydata.org/)
[`matplotlib`](https://matplotlib.org/)

#### file involved  

`pol600withLabel.csv` (data)
`zscore_sort_in_strategies.json` (data sorted in strategies)
`find_feature.ipynb`

#### function and usage:

- `add_feature`


---

## 3. CH/EN Classifier

### 3.1 Data Reader

#### package used:

[`NumPy`](http://www.numpy.org/)  
[`Pandas`](https://pandas.pydata.org/)  

#### file involved:

`ten_models/train_features.pkl`
`ten_models/train.csv`
`ten_models/data_reader.py`

#### function and usage:

1. `data_reader._read`: read data samples  
    
    input:
    - `feature_path`: path to the `pickle` file that stores input feature values, default as `train_features.pkl`
    - `label_path`: path to the `csv` file that stores CH/EN scores of each request, default as `train.csv`
    - `filter_mode` : mode to filter data
      - `""`: do not filter
      - `"percentile"`: only use data in higher/lower `x` percentile in scores, `x` is defined in `filter_val`
      - `"value"`: only use data greater than `x` or less than `-x`, `x` is defined in `filter_val`
    - `filter_val`: used together with `filter_mode`  
    
    output:
    - `np.array` with one sample per row, one feature per column; col `-1` is EN scores, col `-2` is CH scores

2. `data_reader.read`: interface for `_read`

3. `data_reader.k_fold_cross_val`: generator to generate k_fold cross validation data

    input:
    - `k`: number of k
    - `seed`: random seed
    - `feature_path`, `label_path`, `filter_mode`, `filter_val`: for `_read` use  
    
    output:
    - (in each iter) two `np.array`s as training set and validation set


### 3.2 Feed-forward Neural Networks

#### packages used:

[`scikt-learn`](https://scikit-learn.org/)  
[`PyTorch`](https://pytorch.org/tutorials/)  

#### file involved:

`ten_models/data_reader.py` (mentioned above)  
`ten_models/NN_k_fold_cross_val.ipynb`

#### model:

`NNClassifier`:
- init: 
  - `hidden_num`: number of hidden layers, default as `1`
  - `dropout_p`: probability for dropout layer, default as `None` (no dropout)
  - `input_dim`: input feature dimensions, default as `174` (please check the dim of `train_featuers.pkl`)
  - `hidden_dim`: hidden layer dimensions
  - `class_num`: number of class columns (label columns, score columns)
- `forward` and `compute_loss` is defined for `PyTorch` use

#### train and test wrap-up

1. `train_and_test_once`:

    input:
    - `X_train`, `X_test`, `Y_train`, `Y_test`: input features and ground truth labels for classification
    - `hidden_num`, `dropout_p` is for model use (mentioned above)
    - `lr`: learning rate (please check optimizer if the learning rate is used)
    - `epoch_num`: number of training epochs
    - `label_index`: index of label (to choose CH/EN)
    - `debug_mode`: flag if to print debug message

    output:
    - `model`: the trained model
    - `loss`: total validation loss
    - `acc`: validation accuracy

2. `train_and_test_k_fold_cross_val`: train and test under k fold cross validation

    input:
    - `feature_path`, `label_path`, `filter_mode`, `filter_val`, `k`, `seed` for `data_reader.k_fold_cross_val` (mentioned above)
    - `hidden_num`, `dropout_p` for model (mentioned above)
    - `epoch_num`: number of training epochs
    - `debug_mode`: flag if to print debug messages

    output:
    - `best_ch_model`, `best_en_model`: best CH/EN model in k folds

3. `compare_wrap`: calculate the "different results" performance

    input:
    - `model_ch`, `model_en`: model for CH/EN

    output:
    - dictionary contains performance stats


### 3.3 Other ML approaches
@ Chris

