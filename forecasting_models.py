from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import datetime
import data_utilities as dut
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import *

import logging

cmdstanpy_logger = logging.getLogger("cmdstanpy")

cmdstanpy_logger.disabled = True


class BaseForecaster:
    def __init__(self, df, h, additional_df_transformations, data_path, split=TRAIN_EVAL_SPLIT):
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


        self.index_to_dts = {i: dt for i, dt in zip(list(self.initial_df.index), list(self.initial_df["datetimes"]))}

        dts = self.initial_df["datetimes"]
        assert str(type(dts.iloc[0])) == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>"
        assert 0 < split < 1
        self.split = split

        self.check_consecutive_datetimes()

        shortened_path = list(data_path.parts[-4:])
        shortened_path = Path("/".join(shortened_path))
        setattr(df, "source_path", shortened_path)
        self.data_source_path = df.source_path

        self.rolling_predict_rows_to_refit = int(7*24*3600 / dut.get_timedelta(df=self.initial_df))
        self.time_between_rows = dut.get_timedelta(df=self.initial_df)

    def post_init(self):
        self.name = self.__class__.__name__
        self.fn = f"{self.name}_h={self.h}"
        self.data_transformations = self.get_data_transformations()
        self.pipeline = dut.DataPipeline(transforms=self.data_transformations)
        self.transformed_df = self.pipeline(self.initial_df)
        
        n = int(self.split*len(self.transformed_df))
        self.train_df = self.transformed_df.iloc[:n]
        self.test_df = self.transformed_df.iloc[n:] 

        X_df, _ = self.final_df_preprocessing(df=self.transformed_df)
        self.features = list(X_df.columns)

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
        tds = dut.get_timedeltas(df=self.initial_df)
        _td = tds[0]

        assert _td == tds[1]  # verify if the first two tds are the same, required for the clock_shift check
        n_clock_shifts = 0
        for td in tds:
            if td == _td:
                pass
            
            # clock_shift check
            elif td == 3600 + _td or td == -3600 + _td:
                # this occurs during winter/summer time transition
                n_clock_shifts += 1
            
            else:
                raise ValueError("The timedeltas between consecutive steps is inconsistent")
        
            n_years = self.initial_df["datetimes"].iloc[-1].year - self.initial_df["datetimes"].iloc[0].year + 1
            assert n_clock_shifts <= 2*n_years

    def final_df_preprocessing(self, df):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def predict(self, predict_on_test=True, rolling_prediction=False):
        raise NotImplementedError

    def save(self):
        joblib.dump(self, f"{self.fn}.joblib")

    @staticmethod
    def load(fn):
        return joblib.load(fn)

    def logworthy_attributes(self):
        logworthy_attributes = {}
        logworthy_attributes["features"] = self.features
        logworthy_attributes["data_source_path"] = str(self.data_source_path)
        logworthy_attributes["horizon"] = self.h
        logworthy_attributes["model"] = self.name
        logworthy_attributes["train_eval_split"] = self.split
        return logworthy_attributes

class CustomRandomForest(BaseForecaster):
    def __init__(self, df, h, additional_df_transformations, data_path, **kwargs):
        super().__init__(df, h, additional_df_transformations=additional_df_transformations, data_path=data_path)
        self.kwargs = kwargs
        self.model = self.make_model()

        self.ts2row_history_window = 5
        self.ts2row_column_name = "load_profile"
        self.only_fit_using_last_n_weeks = ONLY_FIT_USING_LAST_N_WEEKS

        self.post_init()

    def logworthy_attributes(self):
        logworthy_attributes = super().logworthy_attributes()
        logworthy_attributes = {**logworthy_attributes, **self.kwargs}

        logworthy_attributes["features"] = self.features
        logworthy_attributes["ts2row_history_window"] = self.ts2row_history_window
        logworthy_attributes["only_fit_using_last_n_weeks"] = self.only_fit_using_last_n_weeks
        return logworthy_attributes
        

    def make_model(self):
        return RandomForestRegressor(n_jobs=N_CORES, **self.kwargs)

    def get_base_transformations(self):
        base_transforms = [dut.TimeseriesToRow(column_name=self.ts2row_column_name, 
                history_window=self.ts2row_history_window), 
                dut.DatetimeConversion(),
                dut.AddYesterdaysValue(h=self.h), 
                dut.AddLastWeeksValue(h=self.h), 
                dut.DropNaNs(),
                dut.StandardizeFeatures(train_eval_split=self.split)]
        return base_transforms

    def fit(self):
        X_df, y_series = self.final_df_preprocessing(df=self.train_df)

        X = X_df.to_numpy()
        y = y_series.values

        self.model.fit(X, y)

    def final_df_preprocessing(self, df):
        X_df = df.copy(deep=True)

        y_series = X_df["y"]
        
        X_df = X_df.drop(columns=["datetimes", "y"])

        return X_df, y_series

    def predict(self, predict_on_test=True, rolling_prediction=False):
        if rolling_prediction:
            assert predict_on_test
            self.rolling_predict()
        if predict_on_test:
            X_df, _ = self.final_df_preprocessing(df=self.test_df)
        else:
            X_df, _ = self.final_df_preprocessing(df=self.train_df)

        X = X_df.to_numpy()
        return self.model.predict(X)

    def rolling_predict(self):
        train_X_df, train_y_series = self.final_df_preprocessing(df=self.train_df)
        test_X_df, test_y_series = self.final_df_preprocessing(df=self.test_df)

        n = self.rolling_predict_rows_to_refit
        iterations = np.ceil(len(test_X_df) / n).astype(int)

        extended_train_X_df = train_X_df
        extended_train_y_series = train_y_series
        yhat = []

        new_model = self.model
        for i in range(iterations):

            # grab data corresponding to this rolling window
            sub_test_X_df = test_X_df.iloc[i*n:(i+1)*n]
            sub_test_y_series = test_y_series.iloc[i*n:(i+1)*n]

            # do forecast
            sub_test_X = sub_test_X_df.to_numpy()
            sub_yhat = list(new_model.predict(sub_test_X))
            yhat.extend(sub_yhat)

            # extend the training data
            extended_train_X_df = pd.concat([extended_train_X_df, sub_test_X_df], ignore_index=True)
            extended_train_y_series = pd.concat([extended_train_y_series, sub_test_y_series], ignore_index=True)

            # initialize the new model
            new_model = RandomForestRegressor(**self.kwargs)

            # account for sliding window
            if self.only_fit_using_last_n_weeks > 0:
                rows = int(self.only_fit_using_last_n_weeks*7*24*3600 / self.time_between_rows)
                extended_train_X_df = extended_train_X_df.iloc[-rows:]
                extended_train_y_series = extended_train_y_series.iloc[-rows:]

            # prepare data for the new model
            new_X = extended_train_X_df.to_numpy()
            new_y = extended_train_y_series.values

            # fit the new model
            new_model.fit(new_X, new_y)


        return yhat

class CustomProphet(BaseForecaster):
    def __init__(self, df, h, additional_df_transformations, data_path, **kwargs):
        super().__init__(df, h, additional_df_transformations=additional_df_transformations, data_path=data_path)
        self.only_fit_using_last_n_weeks = ONLY_FIT_USING_LAST_N_WEEKS
        self.post_init()
        self.kwargs = kwargs

        self.model = self.make_model()
        


    def make_model(self):
        model = Prophet(weekly_seasonality=20, daily_seasonality=30, **self.kwargs)
        [model.add_regressor(c, mode="additive") for c in self.transformed_df.columns if c not in ["ds", "y"] and c.startswith("a_")]
        [model.add_regressor(c, mode="multiplicative") for c in self.transformed_df.columns if c not in ["ds", "y"] and c.startswith("m_")]
        return model

    def logworthy_attributes(self):
        logworthy_attributes = super().logworthy_attributes()
        logworthy_attributes = {**logworthy_attributes, **self.kwargs}

        logworthy_attributes["features"] = self.features
        logworthy_attributes["weekly_seasonality"] = self.model.weekly_seasonality
        logworthy_attributes["daily_seasonality"] = self.model.daily_seasonality
        logworthy_attributes["only_fit_using_last_n_weeks"] = self.only_fit_using_last_n_weeks
        logworthy_attributes["rolling_predict_rows_to_refit"] = self.rolling_predict_rows_to_refit

        return logworthy_attributes

    def get_base_transformations(self):
        base_transforms = [dut.AddWeekends(), 
        dut.AddHolidays(),
        dut.DuplicateColumns(prefixes=["a", "m"], exclude=["datetimes", "y"]),
        dut.OnlyFitUsingLastNWeeks(weeks=self.only_fit_using_last_n_weeks),
        dut.StandardizeFeatures(train_eval_split=self.split)]
        return base_transforms

    def fit(self):
        X_df, y_series = self.final_df_preprocessing(df=self.train_df)
        df = X_df.copy(deep=True)
        df["y"] = y_series
        
        self.model.fit(df)

    def final_df_preprocessing(self, df):
        df = df.copy(deep=True)

        if "datetimes" in df.columns:
            df.rename(columns={"datetimes": "ds"}, inplace=True)

        assert "ds" in df.columns
        y_series = df["y"]
        X_df = df.drop(columns="y")
        return X_df, y_series

    def predict(self, predict_on_test=True, rolling_prediction=False):
        if rolling_prediction:
            self.rolling_predict()


        if predict_on_test:
            X_df, y_series = self.final_df_preprocessing(df=self.test_df)
        else:
            X_df, y_series = self.final_df_preprocessing(df=self.train_df)

        df = X_df.copy(deep=True)
        df["y"] = y_series

        forecast = self.model.predict(df)
        yhat = list(forecast["yhat"].values)
        return yhat

    def rolling_predict(self):
        train_X_df, train_y_series = self.final_df_preprocessing(df=self.train_df)
        test_X_df, test_y_series = self.final_df_preprocessing(df=self.test_df)

        n = self.rolling_predict_rows_to_refit
        iterations = np.ceil(len(test_X_df) / n).astype(int)

        extended_train_X_df = train_X_df
        extended_train_y_series = train_y_series
        yhat = []

        new_model = self.model
        for i in range(iterations):

            # grab data corresponding to this rolling window
            sub_test_X_df = test_X_df.iloc[i*n:(i+1)*n]
            sub_test_y_series = test_y_series.iloc[i*n:(i+1)*n]

            # do forecast
            forecast = new_model.predict(sub_test_X_df)
            sub_yhat = list(forecast["yhat"].values)
            yhat.extend(sub_yhat)

            # extend the training data
            extended_train_X_df = pd.concat([extended_train_X_df, sub_test_X_df], ignore_index=True)
            extended_train_y_series = pd.concat([extended_train_y_series, sub_test_y_series], ignore_index=True)

            # initialize the new model
            new_model = self.make_model()

            # prepare data for the new model
            df = extended_train_X_df.copy(deep=True)
            df["y"] = extended_train_y_series

            # account for sliding window
            if self.only_fit_using_last_n_weeks > 0:
                rows = int(self.only_fit_using_last_n_weeks*7*24*3600 / self.time_between_rows)
                df = df.iloc[-rows:]


            # fit the new model
            new_model.fit(df)


        return yhat

class CustomSARIMAX(BaseForecaster):
    def __init__(self, df, h, additional_df_transformations, data_path, **kwargs):
        super().__init__(df, h, additional_df_transformations=additional_df_transformations, data_path=data_path)

        self.kwargs = kwargs
        self.only_fit_using_last_n_weeks = ONLY_FIT_USING_LAST_N_WEEKS

        self.post_init()
        
        self.max_iter = 20
        self.model = self.make_model(df=self.train_df)
        
    
    def make_model(self, df):
        td_between_rows = dut.get_timedelta(self.initial_df)
        samples_per_day = int(24*3600 / td_between_rows)

        X_df, y_series = self.final_df_preprocessing(df=df)
        y = y_series.values
        X = X_df.to_numpy()

        return SARIMAX(endog=y, exog=X, order=(1,1,1), seasonal_order=(0,0,0,samples_per_day), **self.kwargs)

    def logworthy_attributes(self):
        logworthy_attributes = super().logworthy_attributes()
        logworthy_attributes = {**logworthy_attributes, **self.kwargs}

        logworthy_attributes["SARIMAX_max_iter"] = self.max_iter
        logworthy_attributes["SARIMAX_order"] = self.model.order
        logworthy_attributes["SARIMAX_seasonal_order"] = self.model.seasonal_order
        logworthy_attributes["only_fit_using_last_n_weeks"] = self.only_fit_using_last_n_weeks
        logworthy_attributes["rolling_predict_rows_to_refit"] = self.rolling_predict_rows_to_refit
        
        return logworthy_attributes

    def get_base_transformations(self):
        base_transforms = [dut.AddWeekends(), 
                            dut.AddHolidays(),
                            dut.AddLastWeeksValue(h=self.h),
                            dut.AddYesterdaysValue(h=self.h),
                dut.DatetimeConversion(),
                dut.OnlyFitUsingLastNWeeks(weeks=self.only_fit_using_last_n_weeks),
                dut.DropNaNs(),
                dut.StandardizeFeatures(train_eval_split=self.split)]
        return base_transforms

    def fit(self):
        self.fitted_model_parameters = self.model.fit(disp=False, maxiter=self.max_iter)


    def predict(self, predict_on_test=True, rolling_prediction=False):   
        if rolling_prediction:
            return self.rolling_predict()

        if predict_on_test:
            X_df, y_series = self.final_df_preprocessing(df=self.test_df)
            X = X_df.to_numpy()
            y = self.fitted_model_parameters.forecast(steps=len(X), exog=X)
        else:
            assert rolling_prediction is False 

            y = self.fitted_model_parameters.predict()

        return y

    def final_df_preprocessing(self, df):
        df = df.copy(deep=True)
        y_series = df["y"]
        X_df = df.drop(columns=["y"])

        if "datetimes" in X_df.columns:
            X_df = X_df.drop(columns=["datetimes"])

        return X_df, y_series


    def rolling_predict(self):
        train_X_df, train_y_series = self.final_df_preprocessing(df=self.train_df)
        test_X_df, test_y_series = self.final_df_preprocessing(df=self.test_df)

        n = self.rolling_predict_rows_to_refit
        iterations = np.ceil(len(test_X_df) / n).astype(int)

        extended_train_X_df = train_X_df
        extended_train_y_series = train_y_series
        yhat = []

        new_fitted_model = self.fitted_model_parameters
        for i in range(iterations):

            # grab data corresponding to this rolling window
            sub_test_X_df = test_X_df.iloc[i*n:(i+1)*n]
            sub_test_y_series = test_y_series.iloc[i*n:(i+1)*n]

            # do forecast
            sub_test_X = sub_test_X_df.to_numpy()
            sub_yhat = list(new_fitted_model.forecast(steps=len(sub_test_X), exog=sub_test_X))
            yhat.extend(sub_yhat)

            # extend the training data
            extended_train_X_df = pd.concat([extended_train_X_df, sub_test_X_df], ignore_index=True)
            extended_train_y_series = pd.concat([extended_train_y_series, sub_test_y_series], ignore_index=True)

            extended_df = extended_train_X_df.copy(deep=True)
            extended_df["y"] = extended_train_y_series


            # account for sliding window
            if self.only_fit_using_last_n_weeks > 0:
                rows = int(self.only_fit_using_last_n_weeks*7*24*3600 / self.time_between_rows)
                extended_df = extended_df.iloc[-rows:]

            # initialize the new model
            new_model = self.make_model(extended_df)

            # fit the new model
            new_fitted_model = new_model.fit(disp=False, maxiter=self.max_iter)


        return yhat

class CustomSimpleRulesBased(BaseForecaster):
    def __init__(self, df, h, additional_df_transformations, data_path):
        super().__init__(df, h, additional_df_transformations, data_path=data_path)
        self.post_init()

    def get_base_transformations(self):
        base_transforms = [dut.AddLastWeeksValue(h=self.h),
        dut.DropNaNs(),
        dut.OnlyKeepSpecificColumns(columns=["last_weeks_y", "y"]),
        dut.StandardizeFeatures(train_eval_split=self.split)]

        return base_transforms

    def fit(self):
        pass


    def predict(self, predict_on_test=True, rolling_prediction=True):
        if predict_on_test:
            X_df, y_series = self.final_df_preprocessing(df=self.test_df)
        else:
            X_df, y_series = self.final_df_preprocessing(df=self.train_df)

        yhat = X_df["last_weeks_y"].values
        return yhat

    def final_df_preprocessing(self, df):
        y_series = df["y"]
        X_df = df.drop(columns="y")
        
        return X_df, y_series


# class CustomHWES(BaseForecaster):
#     ...

# class CustomVARMAX(BaseForecaster):
#     ...


    
if __name__ == "__main__":
    ...

