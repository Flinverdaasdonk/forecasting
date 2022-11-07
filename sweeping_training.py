import forecasting_models as models
import data_utilities as dut
import visualization_utilities as vut
import logging_utilities as lut
import evaluation_utilities as eut
import training_utilities as tut
from config import *
import time

def sweep_training_dataset_and_log(h):
    adt = []
    for i, data_path in enumerate(tut.yield_train_fns(h=h)):
        print(i)
        df = dut.load_df(data_path)
        m = tut.load_model(df, h, adt, data_path)


        lut.make_and_save_logs(model=m)


if __name__ == "__main__":
    h = HORIZON

    sweep_training_dataset_and_log(h)