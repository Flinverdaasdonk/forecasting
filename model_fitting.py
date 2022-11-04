import forecasting_models as models
import data_utilities as dut
import visualization_utilities as vut
import evaluation_utilities as eut
from config import *

def model_fit_and_eval(model):
    model.fit()

    ### EVAL
    x = eut.get_x(model)
    y = eut.get_y(model)
    yhat = eut.get_yhat(model)

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
        fn = "residential_no_pv\\h={h}_residential_2018_NO_PV_SFH3_2018.csv"
    elif TEST_CATEGORY == "residential_with_pv":
        fn = "residential_with_pv\\h={h}_residential_2018_WITH_PV_SFH13_2018.csv"
    else:
        raise NotImplementedError

    ### LOAD DATA
    path = Path(MAIN_DATA_DIR) / Path("train") / fn
    df = dut.load_df(path)

    # ### INITIALIZE MODEL
    if MODEL_TYPE == "RandomForest":
        m = models.CustomRandomForest(df=df, h=h, additional_df_transformations=adt)
    elif MODEL_TYPE == "SARIMAX":
        m = models.CustomSARIMAX(df=df, h=h, additional_df_transformations=adt)
    elif MODEL_TYPE == "Prophet":
        m = models.CustomProphet(df=df, h=h, additional_df_transformations=adt)

    ### LETSO
    model_fit_and_eval(model=m)
    # ### INITIAL FIT

    m.save()
    # m.fit()

    # ### VISUALIZE RESULTS
    
    # vut.basic_plot(model=m)





