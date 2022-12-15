# Working with the forecasting project
This file helps you get started working with this forecasting project. The goal of the project is to enable forecasting of load profiles from a variety of datasets using a variety of models.

1. I'll start with explaining the file-tree of this project. 
2. Afterwards I'll explain the general workflow of a forecast. 
3. Next I'll refer to a simple script that can be used to do your first forecast. 
3. Lastly I'll refer to some bigger scripts that can be used to do lots of forecasts in one go.

## File-tree structure

This sections gives the filetree structure, as well as a short explanation of each file/folder. This should help you make sense the folder structure

```
datasets/                   <-- contains all datasets
├─ train/                   <-- all training datasets
│  ├─ aggregate/            <-- all horizons of the aggregated household training dataset
│  ├─ industrial/           <-- all horizons and industrial plants for the training dataset
│  ├─ residential_no_pv/    <-- all horizons and households without solar panels for the training dataset
│  ├─ residential_with_pv/  <-- all horizons and households with solar panels for the training dataset
├─ test/                    <-- all test datasets; same structure as /train/
├─ validation/              <-- all validation datasets; same structure as /train/
├─ remainders/              <-- load profiles that are never used
useful notebooks/           <-- Contains lots of (poorly commented) notebooks to do quick prototyping
config.py                   <-- contains script configurations; loaded by every file. Helps to configure number of cores, whether to do a small prototyping test, etc
data_utilities.py           <-- contains utilities for the data processing pipeline that goes inside each model, as well as a couple small functions
deep_learning_utilities.py  <-- Creates the neural network architecture and functions to convert the regular dataframe into a format usable by neural networks
evaluation_utilities.py     <-- Helps you get the targets, predictions, and corresponding datetime for each
forecasting_models.py       <-- Contains the implementations for all forecasting models, the corresponding model parameters, and the data processing pipeline for each model
hyperparameter_tuning.py    <-- No longer relevant; was used to do hyperparameter tuning
logging_utilities.py        <-- Helps you log the results of each model
training.py                 <-- Functions to train any kind of model against loads of datasets
training_utilities.py       <-- Helps with massive training; provides dataloaders for all kinds datasets. These dataloaders iterate over all load profiles in (for example) the training data
visualization_utilities.py  <-- Barely used; gives some rudementary plotting utilities. Instead I recommend logging the results of each model, and later plotting the results of these logs
```

## Generalized forecast workflow

To do a single forecast, the workflow is as follows
1. Load a dataset. This dataset to meet the following criteria
    - typically all dataset filenames start with "h=X_", where 'X' is the forecast horizon
    - Have a datetime column in a format recognizable by pd.to_datetime; probabably "YYYY-mm-dd HH:MM:SS"
    - The timedelta between each datetime should be consistent throughout the dataset
    - A column called "load_profile", which contains the load profile at that datetime
    - A column called "y"; this is the target, for example the load profile in 2 hours
    - No Missing Values in the data
2. Instantiate a model. A model is instantiated with the dataset, and some other information that is used to define the model characteristics, and help with logging later on
    - During model instantiation, it will process the dataset into a format usable by that specific model through its data processing pipeline. 
    - Additionally, will also make a train/test split of the model
3. Fit the model.
    - simply call model.fit(); the model already knows about the data since model instantiation, this wrapper handles everything
4. Predict using the model.
    - call model.predict(); the model will predict in the evaluation part of the data (the data the model hasn't trained on during model.fit())
5. Evaluate.
    - Use the functions inside `evaluation_utilities.py` to grab the predictions and the corresponding datetimes; evaluate however you want using this.

## Making your first forecast

In the file `first_forecast.py` I walk you through a way to do your first forecast in a simple way