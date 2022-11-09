import sys
from config import *
import training_utilities as tut
import data_utilities as dut
import logging_utilities as lut
import time
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

def fit_eval_log(model):
    tic = time.time()

    lut.blockPrint()
    model.fit()
    lut.enablePrint()

    elapsed = time.time() - tic

    logs = {"time_to_fit": elapsed}

    lut.make_and_save_logs(model=model, logs=logs)



def sweep(model_type, horizon):

    if VERBOSE:
        print(f"{model_type}, h={horizon}\n")
    # number of files
    n = len([None for _ in tut.yield_train_fns(h=horizon)])
    adt = [] # additional data transformations

    for i, data_path in enumerate(tut.yield_train_fns(h=horizon)):
        if TINY_TEST:
            # 
            if i % 15 != 0:
                continue

        try:
            if VERBOSE:
                print(f"{i}/{n}")

            df = dut.load_df(data_path)
            m = tut.load_model(df, horizon, adt, data_path)

            fit_eval_log(model=m)


        except Exception as e:
            lut.enablePrint()
            print(f"-----\n ERROR WITH: {model_type} \n at h={horizon} \n for file{data_path}. \n error is: \n {e} \n ------ \n")


def super_sweep():
    model_types = ["RandomForest", "SARIMAX", "Prophet", "RulesBased"]
    horizons = [2, 6]


    for model_type in model_types:
        for horizon in horizons:
            sweep(model_type, horizon)

if __name__ == "__main__":
    VERBOSE = 1

    super_sweep()
    
    # if 'win' in sys.platform: 
    #     model_type = MODEL_TYPE
    #     horizon = HORIZON

    # elif "linux" in sys.platform:
    #     model_type = sys.argv[0]
    #     horizon = sys.argv[1]

    
    



