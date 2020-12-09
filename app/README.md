# EHR Dream Challenge Model Selection Pipeline

This readme outlines the running of the [EHR dream challenge pipeline](https://www.synapse.org/#!Synapse:syn20833371/wiki/600725).

### Prerequisites

The prerequisites are ideally installed into a docker environment using the included Docker file. Or can be installed manually: 

```
catboost
```

## Training

The train.py script has several options to set:

### Models
```model_names = ["ada", "catboost"]``` ([permalink](https://github.com/ivanbrugere/ehrdc/blob/9112f5e3b92ea6f8d8d36d5a79a251e93c68038b/app/train.py#L27)) 


This sets the models to be evaluated. Their definition and parameters are set [here](https://github.com/ivanbrugere/ehrdc/blob/9112f5e3b92ea6f8d8d36d5a79a251e93c68038b/app/model_configs.py#L313).

For now, default model values are inconveniently in the [configs](https://github.com/ivanbrugere/ehrdc/blob/9112f5e3b92ea6f8d8d36d5a79a251e93c68038b/app/model_configs.py#L258), most of these shouldn't need editing, except perhaps:

```config["cv iters"] = 5 ``` ([permalink](https://github.com/ivanbrugere/ehrdc/blob/7d9230f7e0b7326fce9b8c607f518d5513d7ab54/app/model_configs.py#L291))

This adjusts the number of randomized train-test splits to evaluate each model model over. Increasing this yields a more robust estimate of relative model accuracy. Selection is done over the max-mean model (with no variance penalty). 

Note that all models are evaluated on the same train-test split per iteration (rather than sampling a new split per model, per iter). 

## Numpy input

Both pipelines prefer numpy data input, when available. To run on preprocessed numpy input create the following paths:
```
./train/
./test/

```

The mapping for data is defined [here](https://github.com/ivanbrugere/ehrdc/blob/9112f5e3b92ea6f8d8d36d5a79a251e93c68038b/app/model_configs.py#L285)

The "map" are key-value pairs for files and the label value they'll be mapped to. In this instance "alive" is mapped to label 0, and "death" is mapped to label 1.

## Feature importance

I am outputting feature importance scores to:
```
./output/feature_weights.csv
```

This is currently implemented in Adaboost and CatBoost models. Feature importance evaluation need be handled for new models [here](https://github.com/ivanbrugere/ehrdc/blob/82d549bfc67f69373489df1da7ac56cc19061ebb/app/train.py#L70).