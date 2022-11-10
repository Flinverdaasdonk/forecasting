import sys
from config import *
import training_utilities as tut
import data_utilities as dut
import logging_utilities as lut
import time
import warnings
#from pandas.core.common import SettingWithCopyWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=UserWarning)



def fit_eval_log(model):
    tic = time.time()

    lut.blockPrint()
    model.fit()
    lut.enablePrint()

    elapsed = time.time() - tic

    logs = {"time_to_fit": elapsed}

    lut.make_and_save_logs(model=model, logs=logs)



# def sweep(model_type, horizon):

#     if VERBOSE:
#         print(f"{model_type}, h={horizon}\n")
#     # number of files
#     n = len([None for _ in tut.yield_train_fns(h=horizon)])
#     adt = [] # additional data transformations

#     for i, data_path in enumerate(tut.yield_train_fns(h=horizon)):
#         if TINY_TEST:
#             # 
#             if i % 15 != 0:
#                 continue

#         try:
#             if VERBOSE:
#                 print(f"{i}/{n}")

#             df = dut.load_df(data_path)
#             m = tut.load_model(model_type, df, horizon, adt, data_path)

#             fit_eval_log(model=m)


#         except Exception as e:
#             lut.enablePrint()
#             print(f"-----\n ERROR WITH: {m.name} \n at h={horizon} \n for file{data_path}. \n error is: \n {e} \n ------ \n")

def super_sweep_Prophet():
    super_sweep_model_type(model_type="Prophet")

def super_sweep_SARIMAX():
    super_sweep_model_type(model_type="SARIMAX")

def super_sweep_RF():
    super_sweep_model_type(model_type="RandomForest")

def super_sweep_model_type(model_type):
    print("Inside 'super_sweep_model_type'")
    model_types = [model_type]
    horizons = [2, 6]

    adt = []

    for horizon in horizons:
        print("Inside 'horizon for loop'")
        n = len(model_types)*len([None for _ in tut.yield_train_fns(h=horizon)])

        for i, data_path in enumerate(tut.yield_train_fns(h=horizon)):
            print("Inside 'yield_train_fns loop'")
            if TINY_TEST:
                # 
                if i % 15 != 0:
                    continue

            for i2, model_type in enumerate(model_types):
                print("Inside 'model_types loop'")
                try:
                    if VERBOSE:
                        print(f"{i*len(model_types)+i2}/{n}: {model_type}, h={horizon} ")

                    df = dut.load_df(data_path)
                    m = tut.load_model(model_type, df, horizon, adt, data_path)

                    fit_eval_log(model=m)


                except Exception as e:
                    lut.enablePrint()
                    print(f"-----\n ERROR WITH: {m.name} \n at h={horizon} \n for file{data_path}. \n error is: \n {e} \n ------ \n")


if __name__ == "__main__":
    VERBOSE = 1
    arg = int(sys.argv[1])

    print(f"model_type={arg}")

    if arg == 0:
        print("Sweeping Prophet")
        super_sweep_Prophet()
    
    elif arg == 1:
        print("Sweeping RFR")
        super_sweep_RF()

    elif arg == 2:
        print("Sweeping SARIMAX")
        super_sweep_SARIMAX()


    print("Done!")

    
    # if 'win' in sys.platform: 
    #     model_type = MODEL_TYPE
    #     horizon = HORIZON

    # elif "linux" in sys.platform:
    #     model_type = sys.argv[0]
    #     horizon = sys.argv[1]

    
    



