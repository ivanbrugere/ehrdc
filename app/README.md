# EHR Dream Challenge Model Selection Pipeline

This readme outlines the running of the [EHR dream challenge pipeline](https://www.synapse.org/#!Synapse:syn20833371/wiki/600725).

### Prerequisites

The prerequisites are ideally installed into a docker environment using the included Docker file. Or can be installed manually: 

```
catboost
shap
```

## Training

The train.py script has several options to set:

### Models
```model_names = ["logistic", "ada", "catboost"]``` ([permalink](https://github.com/ivanbrugere/ehrdc/blob/a939b4964534c0ffe15811e9928ab114841dde09/app/train.py#L21)) 


This sets the models to be evaluated. Their definition and parameters are set [here](https://github.com/ivanbrugere/ehrdc/blob/457b03eacc506efc0f04ce2ba2e93d17f5e39df3/app/model_configs.py#L13).

For now, default model values are inconveniently in the [configs](https://github.com/ivanbrugere/ehrdc/blob/457b03eacc506efc0f04ce2ba2e93d17f5e39df3/app/model_configs.py#L110), most of these shouldn't need editing, except perhaps:

```config["cv iters"] = 3 ``` ([permalink](https://github.com/ivanbrugere/ehrdc/blob/457b03eacc506efc0f04ce2ba2e93d17f5e39df3/app/model_configs.py#L133))

This adjusts the number of randomized train-test splits to evaluate each model model over. Increasing this yields a more robust estimate of relative model accuracy. Selection is done over the max-mean model (with no variance penalty). 

Note that all models are evaluated on the same train-test split per iteration (rather than sampling a new split per model, per iter). 

## Numpy input

Both pipelines prefer numpy data input, when available. To run on preprocessed numpy input create the following paths:
```
./train/
./test/

```

The mapping for data is defined [here](https://github.com/ivanbrugere/ehrdc/blob/457b03eacc506efc0f04ce2ba2e93d17f5e39df3/app/model_configs.py#L127)

The "map" are key-value pairs for files and the label value they'll be mapped to. In this instance "alive" is mapped to label 0, and "death" is mapped to label 1.

## Feature importance

I am outputting feature importance scores to:
```
./output/feature_weights.csv
```

This is currently implemented in Adaboost, CatBoost, and Logistic Regression models. Feature importance evaluation need be handled for new models [here](https://github.com/ivanbrugere/ehrdc/blob/457b03eacc506efc0f04ce2ba2e93d17f5e39df3/app/train.py#L49).