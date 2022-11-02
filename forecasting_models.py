from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import datetime
import data_utilities as dut
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

class BaseForecaster:
    def __init__(self, df, h, additional_df_transformations, split):
        if additional_df_transformations is None:
            adt = []
        elif not isinstance(additional_df_transformations, list):
            adt = [additional_df_transformations]
        else:
            adt = additional_df_transformations

        self.h = h
        self.additional_df_transformations = adt
        self.initial_df = df.copy(deep=True)
        self.initial_df["datetimes"] = [pd.to_datetime(dt) for dt in self.initial_df["datetimes"]]

        self.index_to_dts = {i: dt for i, dt in zip(list(self.initial_df.index), list(self.initial_df["datetimes"].values))}

        assert 0 < split < 1
        self.split = split

        self.check_consecutive_datetimes()

    def post_init(self):
        self.name = self.__class__.__name__
        self.data_transformations = self.get_data_transformations()
        self.pipeline = dut.DataPipeline(transforms=self.data_transformations)
        self.transformed_df = self.pipeline(self.initial_df)
        
        n = int(self.split*len(self.transformed_df))
        self.train_df = self.transformed_df.iloc[:n]
        self.test_df = self.transformed_df.iloc[n:] 

    def get_data_transformations(self):
        return self.get_base_transformations() + self.additional_df_transformations

    def get_corresponding_dts(self, df):
        idxs = list(df.index)
        dts = [self.index_to_dts[idx] for idx in idxs]
        return dts

    def get_test_dts(self):
        dts = self.get_corresponding_dts(df=self.test_df)
        return dts

    def get_train_dts(self):
        dts = self.get_corresponding_dts(df=self.train_df)
        return dts   

    def check_consecutive_datetimes(self):
        dts = self.initial_df["datetimes"].values
        tds = [(dt1 - dt0).astype('float64')/1e9 for dt1, dt0 in zip(dts[1:], dts[:-1])]

        assert all(td == tds[0] for td in tds), f"The timedeltas between consecutive steps is inconsistent" 


    def fit(self):
        raise NotImplementedError

    def predict(self, df=None):
        raise NotImplementedError


class CustomRandomForest(BaseForecaster):
    def __init__(self, df, h, additional_df_transformations=None, split=0.75, **kwargs):
        super().__init__(df, h, additional_df_transformations=additional_df_transformations, split=split)
        self.model = RandomForestRegressor(**kwargs)

        self.ts2row_history_window = 5
        self.ts2row_column_name = "load_profile"

        self.post_init()

    def get_base_transformations(self):
        base_transforms = [dut.TimeseriesToRow(column_name=self.ts2row_column_name, 
                history_window=self.ts2row_history_window), 
                dut.DatetimeConversion(),
                dut.AddYesterdaysValue(h=self.h), 
                dut.AddLastWeeksValue(h=self.h), 
                dut.DropNaNs()]
        return base_transforms

    def fit(self):
        X, y = self.final_df_preprocessing(df=self.train_df)
        self.model.fit(X, y)

    def final_df_preprocessing(self, df):
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

    def predict(self, df=None):
        if df is None:
            X, _ = self.final_df_preprocessing(df=self.test_df)
        else:
            X, _ = self.final_df_preprocessing(df=df)

        return self.model.predict(X)


class CustomProphet(BaseForecaster):
    def __init__(self, df, h, additional_df_transformations, split=0.75, **kwargs):
        super().__init__(df, h, additional_df_transformations=additional_df_transformations, split=split)
        self.model = Prophet(**kwargs)


        self.post_init()
        [self.model.add_regressor(c, mode="additive") for c in self.transformed_df.columns if c not in ["ds", "y"] and c.startswith("a_")]
        [self.model.add_regressor(c, mode="multiplicative") for c in self.transformed_df.columns if c not in ["ds", "y"] and c.startswith("m_")]


    def get_base_transformations(self):
        base_transforms = [dut.AddWeekends(), 
        dut.AddHolidays(),
        dut.DuplicateColumns(prefixes=["a", "m"], exclude=["datetimes", "y"])]
        return base_transforms

    def fit(self):
        df = self.final_df_preprocessing(df=self.train_df)
        self.model.fit(df)

    def final_df_preprocessing(self, df):
        df = df.copy(deep=True)

        if "datetimes" in df.columns:
            df.rename(columns={"datetimes": "ds"}, inplace=True)

        assert "ds" in df.columns
        return df

    def predict(self, df=None):
        if df is None:
            df = self.final_df_preprocessing(df=self.test_df)
        else:
            df  = self.final_df_preprocessing(df=df)

        if "y" in df.columns:
            df = df.drop(columns=["y"])

        forecast = self.model.predict(df)
        yhat = forecast["yhat"].values
        return yhat


class CustomSARIMAX(BaseForecaster):
    def __init__(self, df, h, additional_df_transformations, split=0.75, **kwargs):
        super().__init__(df, h, additional_df_transformations=additional_df_transformations, split=split)
    
        self.post_init()
        X, y = self.final_df_preprocessing(self.train_df)

        dt0, dt1 = self.initial_df["datetimes"].values[0], self.initial_df["datetimes"].values[0]
        td_between_rows = (dt1-dt0).astype('float64')/1E9  # convert ns to s

        samples_per_day = int(24*3600 / td_between_rows)

        self.model = SARIMAX(endog=y, exog=X, order=(1,1,1), seasonal_order=(0,1,0,samples_per_day), **kwargs)

    def get_base_transformations(self):
        base_transforms = [dut.AddWeekends(), 
                            dut.AddHolidays(),
                            dut.AddLastWeeksValue(h=self.h),
                            dut.AddYesterdaysValue(h=self.h),
                dut.DatetimeConversion(drop_original_column=True),
                dut.DropNaNs()]
        return base_transforms

    def fit(self):
        self.fitted_model_parameters = self.model.fit(disp=False, maxiter=5)


    def predict(self, df=None):
        if df is None:
            X, _ = self.final_df_preprocessing(df=self.test_df)
        else:
            X, _ = self.final_df_preprocessing(df=df)

        return self.fitted_model_parameters.predict(exog=X)

    def final_df_preprocessing(self, df):
        y = df["y"].values
        X = df.drop(columns=["y"]).to_numpy()
        return X, y


class CustomSimpleRulesBased(BaseForecaster):
    def __init__(self, df, h, additional_df_transformations, split=0.75):
        super().__init__(df, h, additional_df_transformations, split)
        self.post_init()

    def get_base_transformations(self):
        base_transforms = [dut.AddLastWeeksValue(h=self.h),
        dut.DropNaNs(),
        dut.OnlyKeepSpecificColumns(columns=["last_weeks_y", "y"])
        ]
        return base_transforms

    def fit(self):
        pass

    def predict(self, df=None):
        if df is None:
            df = self.test_df
        yhat = df["last_weeks_y"].values
        return yhat

    def final_df_preprocessing(self, df):
        X = ...
        y = ...
        return X, y


# class CustomHWES(BaseForecaster):
#     ...

# class CustomVARMAX(BaseForecaster):
#     ...


    
if __name__ == "__main__":
    ...

