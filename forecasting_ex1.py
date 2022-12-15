""" Do a simple forecast using a single model, and a single file.
"""
import pandas as pd
from pathlib import Path
from forecasting_models import CustomRandomForest
import evaluation_utilities as eut
import matplotlib.pyplot as plt

if __name__== "__main__":
    MAIN_DATA_DIR = Path(r"C:\Users\Flin\OneDrive - TU Eindhoven\Flin\Flin\01 - Uni\00_Internship\Nokia\00_Programming\forecasting\datasets")
    HORIZON = 1

    full_file_path = MAIN_DATA_DIR / f"train/industrial/h={HORIZON}_industrial_2016_LG_1.csv"

    # load a dateframe
    df = pd.read_csv(full_file_path)

    # instantiate a model; only 'df' is really important. additional_df_transformations can be added, but the default pipeline is fine. 'h' and 'data_path' are used during logging
    model = CustomRandomForest(df=df, h=HORIZON, additional_df_transformations=[], data_path=full_file_path)

    # fit
    model.fit()

    # get the prediction on the train and test data
    yhat = eut.get_yhat(model)

    # get actual values
    y = eut.get_y(model)

    # get corresponding datetimes
    x = eut.get_x(model)

    # plot the results
    plt.plot(x, y, label="target")
    plt.plot(x, yhat, label='prediction')
    plt.grid()
    plt.legend()
    plt.xlabel("datetime")
    plt.ylabel("Load profile")
    plt.show()

    print("Done")

    