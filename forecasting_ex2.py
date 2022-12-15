import training_utilities as tut
import data_utilities as dut
from config import *

if __name__ == "__main__":
    USE_AGGREGATE = False  # whether to use the aggregated dataset or not
    TIME_COMPRESSION = False  # which time_compression to use; influences the sampling period. tc=False is a sampling period of 15 minutes, tc=4 is a sampling period of 1 hour. 
        # Look at the filenames of the datasets to see which time_compressions are available.
    HORIZON = 2
    MODEL_TYPE = "RandomForest"
    ADDITIONAL_DATA_TRANSFORMATIONS = []
    kwargs = {}

    for i, fn in enumerate(tut.yield_train_fns(h=HORIZON, main_data_dir=MAIN_DATA_DIR, time_compression=TIME_COMPRESSION, use_aggregate=USE_AGGREGATE)):
        df = dut.load_df(fn)

        model = tut.load_model(model_type=MODEL_TYPE, df=df, h=HORIZON, adt=ADDITIONAL_DATA_TRANSFORMATIONS, data_path=fn, **kwargs)

        # fit the model, evaluate the model, and log the results
        tut.fit_eval_log(model)

        if i == 0:
            break



    print("Done")