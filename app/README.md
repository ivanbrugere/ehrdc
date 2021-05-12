# Synthetic EHR Data Evaluation Pipeline

This readme outlines the running of the evaluation pipeline for generative EHR models, on binary or continuous features, for binary prediction targets.

### Prerequisites

This pipeline depends on the following non-standard libraries: 

```
catboost
shap
```
## Training

The train.py script has several options to set:

### Models
```model_names = ["logistic", "ada", "catboost"]``` ([permalink](https://github.com/ivanbrugere/ehrdc/blob/ffb53389566b044c91ae5422f1e6ef428ccf7687/app/train.py#L12)) 


The [get_baseline_cv_configs](https://github.com/ivanbrugere/ehrdc/blob/ffb53389566b044c91ae5422f1e6ef428ccf7687/app/model_configs.py#L12) defines the models and their hyper-parameters to be evaluated according to the passed 'model_names'

The [get_base_config](https://github.com/ivanbrugere/ehrdc/blob/ffb53389566b044c91ae5422f1e6ef428ccf7687/app/model_configs.py#L102) defines paths and other pipeline parameters, two important defaults are:


```config["k folds"] = 10 ``` ([permalink](https://github.com/ivanbrugere/ehrdc/blob/ffb53389566b044c91ae5422f1e6ef428ccf7687/app/model_configs.py#L119))


This adjusts the number of folds in k-fold validation.  

### Paths

```config["train npy"] = {"path": os.path.join("..", "train", ""), "map": {"negative.npy": 0, "positive.npy": 1}, "fields": {"data": "x", "labels":"y"}}```

*  This [line](https://github.com/ivanbrugere/ehrdc/blob/ffb53389566b044c91ae5422f1e6ef428ccf7687/app/model_configs.py#L117) specifies the paths to training data. 
*  The 'map' key specifies two numpy matrices found in 'path.' These matrices are of size [MxF] and [NxF] for negative and positive class instances, and labels are created with the associated value within the map (e.g. negative.npy → 0). 
*  The 'field' key specifies the output dictionary for the data and labels, e.g. d["x"] and d["y"] in the above example. (This shouldn't need changing)

In summary, the pipeline expects the following files:
```
../train/negative.npy   # [* x F] numpy matrix
../train/positive.npy   # [* x F] numpy matrix
../test/negative.npy    # [* x F] numpy matrix
../test/positive.npy    # [* x F] numpy matrix
```

## Feature importance

This pipeline generates feature importance. The importance is given by:

* Logistic regression: The learned feature weights of the model
* Adaboost: the internal Gini importance from the sklearn library
* Catboost: the mean shap value across *test* instances. 

See [here](https://github.com/ivanbrugere/ehrdc/blob/ffb53389566b044c91ae5422f1e6ef428ccf7687/app/models.py#L51).

## Output

This pipeline outputs the following in the specified output path:

* Models (models.gz): List of best-performing models in each fold (List[Object], [1 x k])
* Feature importance (feature_weights.csv): CSV of the feature importance associated with the above k models ([F x k])
* Prediction scores (predictions.csv): CSV of the test set evaluation of each k model ([$N_{test}$ x k])