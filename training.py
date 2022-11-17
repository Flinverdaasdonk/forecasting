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



def super_sweep(model_type):
    
    n_files = [len([None for _ in tut.generic_yield_fns(h=h, specific_folder_name=TRAIN_OR_TEST)]) for h in HORIZONS]

    iterations = 1
    for n in n_files:
        iterations *= n

    if TINY_TEST:
        skip_n = 15
        iterations = iterations // skip_n

    adt = []

    for horizon in HORIZONS:
        for i, data_path in enumerate(tut.generic_yield_fns(h=horizon, specific_folder_name=TRAIN_OR_TEST)):
            if TINY_TEST and i % skip_n != 0:
                continue
                                
            if VERBOSE:
                print(f"{i}/{iterations}: {model_type}, h={horizon}")

            try:

                df = dut.load_df(data_path)
                m = tut.load_model(model_type, df, horizon, adt, data_path)

                fit_eval_log(model=m)

            except Exception as e:
                lut.enablePrint()
                print(f"-----\n ERROR WITH: {m.name} \n at h={horizon} \n for file{data_path}. \n error is: \n {e} \n ------ \n")


def single_file(model_type):

    data_path = next(iter(((Path(MAIN_DATA_DIR) / TRAIN_OR_TEST) / TEST_CATEGORY).iterdir()))

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
        parser = argparse.ArgumentParser()
        parser.add_argument("model_type", type=str, choices=["RandomForest", "SARIMAX", "LSTM", "Prophet", "LastWeeks", "Yesterdays"])
        parser.add_argument("--verbose", type=bool, default=True)
        parser.add_argument("--test_single_file", type=bool, default=False)

        args = parser.parse_args()

        model_type = args.model_type
        test_single_file = args.test_single_file

        VERBOSE = args.verbose

    else:
        print("Running from script")
        model_type = "RandomForest"
        test_single_file = True
        VERBOSE = True

    if test_single_file:
        print(f"{time.ctime()} - Starting single_file test with {model_type} and tiny_test={TINY_TEST}")
        single_file(model_type)
    else:
        print(f"{time.ctime()} - Starting super_sweep test with {model_type} and tiny_test={TINY_TEST}")
        super_sweep(model_type)


    

    # super_sweep_model_type("LSTM")


    # print("Done!")

    
    # if 'win' in sys.platform: 
    #     model_type = MODEL_TYPE
    #     horizon = HORIZON

    # elif "linux" in sys.platform:
    #     model_type = sys.argv[0]
    #     horizon = sys.argv[1]

    
    



