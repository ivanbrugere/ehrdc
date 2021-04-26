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
```model_names = ["logistic", "ada", "catboost"]``` ([permalink](https://github.com/ivanbrugere/ehrdc/blob/693bfe18a2c1b5f48cf758528185aec597846b75/app/train.py#L21)) 


The [get_baseline_cv_configs](https://github.com/ivanbrugere/ehrdc/blob/693bfe18a2c1b5f48cf758528185aec597846b75/app/model_configs.py#L13) defines the models and their hyper-parameters to be evaluated according to the passed 'model_names'

The [get_base_config](https://github.com/ivanbrugere/ehrdc/blob/693bfe18a2c1b5f48cf758528185aec597846b75/app/model_configs.py#L110) defines paths and other pipeline parameters, two important defaults are:


```config["cv iters"] = 3 ``` ([permalink](https://github.com/ivanbrugere/ehrdc/blob/457b03eacc506efc0f04ce2ba2e93d17f5e39df3/app/model_configs.py#L133))


This adjusts the number of randomized train-test splits to evaluate each model model over. Increasing this yields a more robust estimate of relative model accuracy. Selection is done over the maximum-mean model. 

Note that all models are evaluated on the same train-test split per iteration (rather than sampling a new split per model, per iter). 

### Paths

```config["train npy"] = {"path": os.path.join("..", "train", ""), "map": {"negative.npy": 0, "positive.npy": 1}, "fields": {"data": "x", "labels":"y"}}```

*  This [line](https://github.com/ivanbrugere/ehrdc/blob/693bfe18a2c1b5f48cf758528185aec597846b75/app/model_configs.py#L127) specifies the paths to training data. 
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

I am outputting feature importance scores to:
```
./output/feature_weights.csv
```

This is currently implemented in Adaboost, CatBoost, and Logistic Regression models. Feature importance evaluation need be handled for new models [here](https://github.com/ivanbrugere/ehrdc/blob/457b03eacc506efc0f04ce2ba2e93d17f5e39df3/app/train.py#L49).