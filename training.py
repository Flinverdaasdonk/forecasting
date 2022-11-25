import sys
from config import *
import training_utilities as tut
import data_utilities as dut
import logging_utilities as lut
import argparse
import time
import warnings
from pathlib import Path
#from pandas.core.common import SettingWithCopyWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=UserWarning)



def fit_eval_log(model):
    tic = time.time()
    logs = {}

    logs["time_before_fit"] = repr(time.ctime())

    lut.blockPrint()
    model.fit()
    lut.enablePrint()

    logs["time_to_fit"] = time.time() - tic

    lut.make_and_save_logs(model=model, logs=logs)


def hyperparameter_sweep(**kwargs):
    model_type = "RandomForest"
    SKIP_EVERY_N_FILES = 5  # speeds up sweeping

    adt = []
    total_mse = 0

    tot = 0
    for horizon in HORIZONS:
        print(f"------- h={horizon} ------")
        for i, data_path in enumerate(tut.yield_train_fns(h=horizon)):
            if i % SKIP_EVERY_N_FILES != 0 or "residential_2018_NO_PV_SFH10_2018" in data_path.stem:
                continue
            

            df = dut.load_df(data_path)
            m = tut.load_model(model_type, df, horizon, adt, data_path, **kwargs)

            m.fit()

            yhat_test = list(m.predict(predict_on_test=True, rolling_prediction=ROLLING_PREDICTION))
            y_test = list((m.transformed_df["y"].values))[-len(yhat_test):]

            mse = mean_squared_error(y_test, yhat_test)

            total_mse += mse

            tot += 1

 

    return total_mse
            

def super_sweep(model_type, time_compression, aggregate):

    if model_type == "RandomForest":
        kwargs = {"max_features": 0.4, "max_samples":0.4, "n_estimators": 400, "history_depth": 8}
    else:
        kwargs = {}

    iterations = sum([len([None for _ in tut.generic_yield_fns(h=h, specific_folder_name=EVAL_TYPE, time_compression=time_compression, use_aggregate=aggregate)]) for h in HORIZONS])

    if TINY_TEST:
        skip_n = 15
        iterations = iterations // skip_n

    adt = []

    for horizon in HORIZONS:
        for i, data_path in enumerate(tut.generic_yield_fns(h=horizon, specific_folder_name=EVAL_TYPE)):
            if TINY_TEST and i % skip_n != 0:
                continue
                                
            if VERBOSE:
                print(f"{i}/{iterations}: {model_type}, h={horizon}")

            try:
                df = dut.load_df(data_path)
                m = tut.load_model(model_type, df, horizon, adt, data_path, **kwargs)

                fit_eval_log(model=m)

            except Exception as e:
                lut.enablePrint()
                print(f"-----\n ERROR WITH: {m.name} \n at h={horizon} \n for file{data_path}. \n error is: \n {e} \n ------ \n")


def single_file(model_type, time_compression, aggregate):
    H=18
    
    data_path = next(iter(tut.generic_yield_fns(h=H, specific_folder_name=EVAL_TYPE, time_compression=time_compression, use_aggregate=aggregate)))

    if VERBOSE:
        print(f"Single sweep: {model_type}, h={HORIZON}, fn={('/').join(data_path.parts[-3:])}")
    try:

        df = dut.load_df(data_path)
        m = tut.load_model(model_type, df, HORIZON, [], data_path)

        fit_eval_log(model=m)

    except Exception as e:
        lut.enablePrint()
        print(f"-----\n ERROR WITH: {m.name} \n at h={HORIZON} \n for file{data_path}. \n error is: \n {e} \n ------ \n")


if __name__ == "__main__":

    CLI = False

    if CLI:
        print("Running from CLI")
        parser = argparse.ArgumentParser()
        parser.add_argument("model_type", type=str, choices=["RandomForest", "SARIMAX", "LSTM", "Prophet", "LastWeeks", "Yesterdays"])
        parser.add_argument("--verbose", type=bool, default=True)
        parser.add_argument("--test_single_file", type=bool, default=False)
        parser.add_argument("--aggr", type=bool, default=False)
        parser.add_argument("--tc", type=bool, default=False)

        args = parser.parse_args()

        model_type = args.model_type
        test_single_file = args.test_single_file
        aggregate = args.aggr
        time_compression = args.tc

        VERBOSE = args.verbose

    else:
        print("Running from script")
        model_type = "RandomForest"
        test_single_file = False
        VERBOSE = True
        aggregate = True
        time_compression = False

    if test_single_file:
        print(f"{time.ctime()} - Starting single_file test with {model_type} and tiny_test={TINY_TEST}")
        single_file(model_type, time_compression=time_compression, aggregate=aggregate)
    else:
        print(f"{time.ctime()} - Starting super_sweep: {model_type}. tiny_test={TINY_TEST}, aggregate={aggregate}, time_compression={time_compression}")
        super_sweep(model_type, time_compression=time_compression, aggregate=aggregate)

    print("------------------------ \n FINISHED WITH ALL \n -----------------------------------------")
    



