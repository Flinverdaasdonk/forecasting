import forecasting_models as models
import data_utilities as dut
import visualization_utilities as vut
from config import *

def make_train_test_split(df, split):
    assert 0 <= split <= 1
    n = int(len(df)*split)
    train_df = df.iloc[:n].copy(deep=True)
    test_df = df.iloc[n:].copy(deep=True)

    return train_df, test_df

if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    import pandas as pd


    ### CONSTANTS
    h = HORIZON
    adt = []

    ### GET FILENAME
    if TINY_TEST_CATEGORY == "industrial":
        fn = f"industrial\\h={h}_industrial_2016_LG_1.csv"
    elif TINY_TEST_CATEGORY == "residential_no_pv":
        fn = "residential_no_pv\\h={h}_residential_2018_NO_PV_SFH3_2018.csv"
    elif TINY_TEST_CATEGORY == "residential_with_pv":
        fn = "residential_with_pv\\h={h}_residential_2018_WITH_PV_SFH13_2018.csv"
    else:
        raise NotImplementedError

    ### LOAD DATA
    path = Path(MAIN_DATA_DIR) / Path("train") / fn
    df = pd.read_csv(path)
    df = df.iloc[4*24*7*2:4*24*7*4] if TINY_TEST else df


    # ### INITIALIZE MODEL
    if MODEL_TYPE == "RandomForest":
        m = models.CustomRandomForest(df=df, additional_data_transformations=adt)
    elif MODEL_TYPE == "SARIMAX":
        m = models.CustomSARIMAX(df=df, h=h, additional_df_transformations=adt)
    elif MODEL_TYPE == "Prophet":
        m = models.CustomProphet(df=df, h=h, additional_df_transformations=adt)


    ### INITIAL FIT
    m.fit()

    ### VISUALIZE RESULTS
    
    vut.basic_plot(model=m)





