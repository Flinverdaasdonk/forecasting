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
    tiny_test = False
    residential = False

    ### PREPARE DATA
    h = 2
    usable_data_folder = Path(r"C:\Users\Flin\OneDrive - TU Eindhoven\Flin\Flin\01 - Uni\00_Internship\Nokia\00_Programming\forecasting\datasets\train")

    fn = f"residential_no_pv\\h={h}_residential_2018_NO_PV_SFH3_2018.csv" if residential else f"industrial\\h={h}_industrial_2016_LG_1.csv"
    path = usable_data_folder / fn

    df = pd.read_csv(path)
    df = df.iloc[:4*24*7*4] if tiny_test else df


    # ### INITIALIZE MODEL
    adt = []
    # m = models.CustomRandomForest(df=df, additional_data_transformations=adt)  # 
    m = models.CustomRandomForest(df=df, h=h, additional_df_transformations=adt)
    m.fit()

    ### VISUALIZE RESULTS
    fig, axs = plt.subplots(2, figsize=(8, 8))

    vut.plot_predictions(ax=axs[0], model=m)
    vut.plot_prediction_error(ax=axs[1], model=m)

    plt.show()


    if isinstance(m, models.CustomRandomForest):

        # plot feature importance
        features = m.features
        del features[features.index("y")]
        if "datetimes" in m.transformed_data.columns:
            del features[features.index("datetimes")]

        importances = m.model.feature_importances_

        assert len(features) == len(importances)
        d = {f:i for f, i in zip(features, importances)}

        plt.barh(features, importances)
        plt.show()


        print("Done!")

