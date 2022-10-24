from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import data_utilities as dut
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class BaseForecaster:
    def __init__(self, df, additional_data_transformations, split):
        if additional_data_transformations is None:
            adt = []
        elif not isinstance(additional_data_transformations, list):
            adt = [additional_data_transformations]
        else:
            adt = additional_data_transformations

        self.additional_data_transformations = adt
        self.initial_data = df.copy(deep=True)
        self.initial_data["datetimes"] = pd.to_datetime(self.initial_data["datetimes"])

        self.index_to_dts = {i: dt for i, dt in zip(list(self.initial_data.index), list(self.initial_data["datetimes"].values))}

        assert 0 < split <= 1
        self.split = split

    def post_init(self):
        self.name = self.__class__.__name__
        self.data_transformations = self.get_data_transformations()
        self.pipeline = dut.DataPipeline(transforms=self.data_transformations)
        self.transformed_data = self.pipeline(self.initial_data)
        
        n = int(self.split*len(self.transformed_data))
        self.train_data = self.transformed_data.iloc[:n]
        self.test_data = self.transformed_data.iloc[n:] 

    def get_data_transformations(self):
        return self.get_base_transformations() + self.additional_data_transformations

    def fit(self):
        raise NotImplementedError

    def predict(self, X=None):
        raise NotImplementedError

    def get_corresponding_dts(self, df):
        idxs = list(df.index)
        dts = [self.index_to_dts[idx] for idx in idxs]
        return dts

    def get_test_dts(self):
        dts = self.get_corresponding_dts(df=self.test_data)
        return dts

    def get_train_dts(self):
        dts = self.get_corresponding_dts(df=self.train_data)
        return dts     


class CustomRandomForest(BaseForecaster):
    def __init__(self, df, additional_data_transformations=None, split=0.75, **kwargs):
        super().__init__(df, additional_data_transformations=additional_data_transformations, split=split)
        self.model = RandomForestRegressor(**kwargs)

        self.ts2row_history_window = 5
        self.ts2row_column_name = "load_profile"

        self.post_init()

    def get_base_transformations(self):
        base_transforms = [dut.TimeseriesToRow(column_name=self.ts2row_column_name, 
                history_window=self.ts2row_history_window), 
                dut.DatetimeConversion()]
        return base_transforms

    def fit(self):
        X, y = self.final_preprocessing_data(df=self.train_data)
        self.model.fit(X, y)

    def final_preprocessing_data(self, df):
        X = df.copy(deep=True)
        if "y" in X.columns:
            y = X["y"].values
            X = X.drop(columns=["y"])
        else:
            y = None

        if "datetimes" in X.columns:
            X = X.drop(columns=["datetimes"])

        X = X.to_numpy()
        return X, y

    def predict(self, data=None):
        if data is None:
            X, _ = self.final_preprocessing_data(df=self.test_data)
        else:
            X, _ = self.final_preprocessing_data(df=data)

        return self.model.predict(X)


class CustomProphet(BaseForecaster):
    def __init__(self, df, additional_data_transformations, split=0.75, **kwargs):
        super().__init__(df, additional_data_transformations=additional_data_transformations, split=split)
        self.model = Prophet(**kwargs)


        self.post_init()
        [self.model.add_regressor(c, mode="additive") for c in self.transformed_data.columns if c not in ["ds", "y"] and c.startswith("a_")]
        [self.model.add_regressor(c, mode="multiplicative") for c in self.transformed_data.columns if c not in ["ds", "y"] and c.startswith("m_")]


    def get_base_transformations(self):
        base_transforms = [dut.AddWeekends(), 
        dut.AddHolidays(),
        dut.DuplicateColumns(prefixes=["a", "m"], exclude=["datetimes", "y"])]
        return base_transforms

    def fit(self):
        df = self.final_preprocessing_data(df=self.train_data)
        self.model.fit(df)

    def final_preprocessing_data(self, df):
        df = df.copy(deep=True)

        if "datetimes" in df.columns:
            df.rename(columns={"datetimes": "ds"}, inplace=True)

        assert "ds" in df.columns
        return df

    def predict(self, data=None):
        if data is None:
            df = self.final_preprocessing_data(df=self.test_data)
        else:
            df  = self.final_preprocessing_data(df=data)

        if "y" in df.columns:
            df = df.drop(columns=["y"])

        return self.model.predict(df)


    
    
if __name__ == "__main__":
    ...

