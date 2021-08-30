# Synthetic EHR Data Evaluation Pipeline

This readme outlines the running of the evaluation pipeline for generative EHR models, on binary or continuous features, for binary prediction targets.

### Prerequisites

This pipeline depends on the following non-standard libraries: 

```
catboost
shap
```
## Training

The run.py script has several options to set:

### Models
```model_names = ["logistic", "ada", "catboost"]``` ([permalink](https://github.com/ivanbrugere/ehrdc/blob/77d902334c17684b8911db1d4b5f58f49cd89a77/app/run.py#L15)) 


The [get_baseline_cv_configs](https://github.com/ivanbrugere/ehrdc/blob/77d902334c17684b8911db1d4b5f58f49cd89a77/app/model_configs.py#L12) defines the models and their hyper-parameters to be evaluated according to the passed 'model_names'

The [get_base_config](https://github.com/ivanbrugere/ehrdc/blob/77d902334c17684b8911db1d4b5f58f49cd89a77/app/model_configs.py#L102) defines paths and other pipeline parameters, two important defaults are:


```config["k folds"] = 10 ``` ([permalink](https://github.com/ivanbrugere/ehrdc/blob/77d902334c17684b8911db1d4b5f58f49cd89a77/app/model_configs.py#L122))


This adjusts the number of folds in k-fold validation.  

### Paths

```config["train npy"] = {"path": os.path.join("..", "train", ""), "map": {"negative.npy": 0, "positive.npy": 1}, "fields": {"data": "x", "labels":"y"}}```

*  This [line](https://github.com/ivanbrugere/ehrdc/blob/77d902334c17684b8911db1d4b5f58f49cd89a77/app/model_configs.py#L119) specifies the paths to training data. 
*  The 'map' key specifies two numpy matrices found in 'path.' These matrices are of size [MxF] and [NxF] for negative and positive class instances, and labels are created with the associated value within the map (e.g. negative.npy → 0). 
*  The 'field' key specifies the output dictionary for the data and labels, e.g. d["x"] and d["y"] in the above example. (This shouldn't need changing)

In summary, the pipeline expects the following input files:
```
../train/negative.npy   # [N_train x F] numpy matrix
../train/positive.npy   # [N_train x F] numpy matrix
../test/negative.npy    # [N_test x F] numpy matrix
../test/positive.npy    # [N_test x F] numpy matrix
```

## Feature importance

This pipeline generates feature importance. The importance is given by:

* Logistic regression: The learned feature weights of the model
* Adaboost: the internal Gini importance from the sklearn library
* Catboost: the mean shap value across *test* instances. 

See [here](https://github.com/ivanbrugere/ehrdc/blob/77d902334c17684b8911db1d4b5f58f49cd89a77/app/models.py#L52).

## Model re-use

This pipeline allows specifying testing on an existing model. If a model file exists in the specified output path, the model will be loaded and testing will occur on the provided test files:

``
config_params = {"load_model": True,
                 "output_path": None,
                 "k_folds": 10}
`` ([permalink](https://github.com/ivanbrugere/ehrdc/blob/77d902334c17684b8911db1d4b5f58f49cd89a77/app/run.py#L17-L16))

## Output

By default, this pipeline generates an output directory in the pattern:

``../output_["%Y%m%d%H%M%S"]``

The following are written in this directory:

* Models (models.gz): The serialized best model (Dict)
* Feature importance (feature_weights.csv): CSV of the feature importance of the selected model evaluated in test ([F x 1])
* Prediction scores (predictions.csv): CSV of the test set evaluation of the selected model ([N_test x 1])
* SHAP matrix (shap_matrix.gz): the serialized matrix of all SHAP measurements ([N_test x 1])
* Model performance list (performance.gz), a list of model performances for each fold (List[N_models])

In summary, the pipeline outputs the following files (with example timestamp):
```
../output_20210830135056/model.gz   # Dict
../output_20210830135056/feature_weights.csv   # [F x 1] numpy matrix
../output_20210830135056/predictions.csv    # [N_test x 1] numpy matrix
../output_20210830135056/performance.gz    # List[N_models]
```