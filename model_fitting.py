import forecasting_models as models
import data_utilities as dut
import visualization_utilities as vut
import logging_utilities as lut
import evaluation_utilities as eut
from config import *
import time

def evaluate(model, logs=None):
    x = eut.get_x(model)
    y = eut.get_y(model)

    tic = time.time()
    yhat = eut.get_yhat(model)
    elapsed = time.time() - tic

    if logs is not None:
        logs["time_to_inference"] = elapsed

    return x, y, yhat, logs

def model_fit_and_eval(model):
    model.fit()

    x, y, yhat, _ = evaluate(model)


    res = {"x": x, "y": y, "yhat": yhat}
    fig, ax = plt.subplots()
    vut.plot_predictions(ax, res, model)

    plt.show()

def fit_eval_log(model):
    logs = {}

    tic = time.time()
    model.fit()
    elapsed = time.time() - tic
    print(f"time_to_fit: {elapsed}")
    logs = {"time_to_fit": elapsed}
    x, y, yhat, logs = evaluate(model, logs)

    logs["x"] = [str(_x) for _x in x]
    logs["y"] = [float(_y) for _y in y]
    logs["yhat"] = [float(_yh) for _yh in yhat]

    lut.make_and_save_logs(model, logs)

    res = {"x": x, "y": y, "yhat": yhat}

    fig, ax = plt.subplots()
    vut.plot_predictions(ax, res, model)

    plt.show()
    




if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    import pandas as pd


    ### CONSTANTS
    h = HORIZON
    adt = []

    ### GET FILENAME
    if TEST_CATEGORY == "industrial":
        fn = f"industrial\\h={h}_industrial_2016_LG_1.csv"
    elif TEST_CATEGORY == "residential_no_pv":
        fn = f"residential_no_pv\\h={h}_residential_2018_NO_PV_SFH3_2018.csv"
    elif TEST_CATEGORY == "residential_with_pv":
        fn = f"residential_with_pv\\h={h}_residential_2018_WITH_PV_SFH13_2018.csv"
    else:
        raise NotImplementedError

    ### LOAD DATA
    data_path = Path(MAIN_DATA_DIR) / Path("train") / fn
    df = dut.load_df(data_path)

    # ### INITIALIZE MODEL
    if MODEL_TYPE == "RandomForest":
        m = models.CustomRandomForest(df=df, h=h, additional_df_transformations=adt, data_path=data_path)
    elif MODEL_TYPE == "SARIMAX":
        m = models.CustomSARIMAX(df=df, h=h, additional_df_transformations=adt, data_path=data_path)
    elif MODEL_TYPE == "Prophet":
        m = models.CustomProphet(df=df, h=h, additional_df_transformations=adt, data_path=data_path)

    ### LETSO
    fit_eval_log(model=m)
    # ### INITIAL FIT


    # m.fit()

    # ### VISUALIZE RESULTS
    
    # vut.basic_plot(model=m)





