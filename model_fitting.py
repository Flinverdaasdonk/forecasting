import forecasting_models as models
import data_utilities as dut
import visualization_utilities as vut

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
    
    print("Start forecasting")
    tiny_test = True

    ### PREPARE DATA
    usable_data_folder = Path(r"C:\Users\Flin\OneDrive - TU Eindhoven\Flin\Flin\01 - Uni\00_Internship\Nokia\00_Programming\data_preparation\data\usable")
    fn = r"residential\h=2_residential_2018_NO_PV_SFH34_2018.csv"  # r"industrial\h=2_industrial_2016_LG_1.csv"
    path = usable_data_folder / fn

    df = pd.read_csv(path)
    df = df.iloc[:100] if tiny_test else df


    ### INITIALIZE MODEL
    adt = []
    m = models.CustomProphet(df=df, additional_data_transformations=adt)  # 
    m.fit()


    ### VISUALIZE RESULTS
    fig, axs = plt.subplots(2, figsize=(8, 8))

    vut.plot_predictions(ax=axs[0], model=m)
    vut.plot_prediction_error(ax=axs[1], model=m)

    plt.show()



    # # plot feature importance
    # features = list(crf.transformed_data.columns)
    # del features[features.index("y")]
    # if "datetimes" in crf.transformed_data.columns:
    #     del features[features.index("datetimes")]

    # importances = crf.model.feature_importances_

    # assert len(features) == len(importances)
    # d = {f:i for f, i in zip(features, importances)}

    # plt.barh(features, importances)
    # plt.show()

    # assert len(y) == len(yhat)
    

    # print("Done!")

