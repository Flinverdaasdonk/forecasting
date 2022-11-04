import numpy as np
import pandas as pd
import datetime
from config import *
import holidays

class DataPipeline:
    """Container for all required transformations for a specific forecasting model

    Args:
        transforms; list: the transforms for all data
        
    Returns:
        None
    
    Raises:
        No errors

    """
    def __init__(self, transforms=None):
        if transforms is not None and not isinstance(transforms, list):
            transforms = [transforms]

        self.transforms = [] if transforms is None else transforms

    def __call__(self, df):
        return self.apply_df_transformations(transforms=self.transforms, df=df)

    @staticmethod
    def apply_df_transformations(transforms, df):
        transformed_df = df.copy(deep=True)
        for tf in transforms:
            transformed_df = tf(transformed_df)

        return transformed_df


class Transform:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, df):
        raise NotImplementedError


class TimeseriesToRow(Transform):
    """ Takes in a timeseries, and converts it into a multivariate dataset based on the input data.
    E.g. for a window of 2:
    [a,b,c,d]=[w,x,y,z] -> [[a,b],[b,c], [c,d]] = [x, y, z]
    
    """
    def __init__(self, column_name, history_window):
        super().__init__(column_name=column_name, history_window=history_window)

        assert isinstance(column_name, str)
        assert isinstance(history_window, int)
        assert history_window > 0

    def __call__(self, df):
        # grab the column from the df we want to offset, e.g. [a, b, c, d]
        series = list(df[self.column_name].values)

        # convert the series to a list of lists, where each sub-list contains the history window for that timestep
        # (see docstring example). e.g. [[a, b], [b, c], [c, d]]
        timeseries_matrix = [series[i:i+self.history_window] for i in range(len(series) - self.history_window + 1)]
        timeseries_matrix = np.array(timeseries_matrix)

        """ NOTE:
        The matrix is currently a (len(df)-history_window+1, history_window) matrix.
        The LAST column is the most RECENT entry for the value we're trying to predict (since the input timeseries goes forward in time)
        """ 

        # Remove the first entries from the dataframe, to account for the lost entries (again, see docstring example)
        shortened_df = df.copy(deep=True).iloc[self.history_window-1:]  # e.g [a, b, c, d]=[w,x,y,z] -> [b,c,d]=[x,y,z]; remember that target column [w,x,y,z] is IN df

        # FInd out at which column index the original column was placed
        col_idx = list(df.columns).index(self.column_name)
        
        # remove the original column
        shortened_df.drop(columns=[self.column_name], inplace=True)

        # for each step along the history
        for i in range(self.history_window):

            name = f"{self.column_name}"
            if i > 0:
                name += f"-{i}"


            # grab the appropriate values from the ts_mat; 
            # note; last column 
            mat_idx = (self.history_window - 1) - i
            values = timeseries_matrix[:, mat_idx]

            shortened_df.insert(loc=col_idx+i, column=name, value=values)

        return shortened_df


class DatetimeConversion(Transform):
    def __init__(self,
                 dt_column="datetimes", 
                 yearly="int", 
                 monthly="sines", 
                 daily=False, 
                 hourly="sines", 
                 add_cosines=True):

        super().__init__(dt_column=dt_column, 
                        yearly=yearly, 
                        monthly=monthly, 
                        daily=daily, 
                        hourly=hourly, 
                        add_cosines=add_cosines)

        assert yearly in ["int", False]
        assert monthly in ["sines", False]
        assert daily in ["sines", False]
        assert hourly in ["sines", False]
        assert isinstance(add_cosines, bool)

    def __call__(self, df):
        dts = df[self.dt_column] # grab dtimes

        dt_df = df.copy(deep=True).iloc[:, :0]  # make deepcopy and only grab index 

        dt_df = DatetimeConversion.add_converted_dt_column(dts=dts, dt_df=dt_df, arg=self.yearly, name="year", base=None, add_cosines=self.add_cosines)
        dt_df = DatetimeConversion.add_converted_dt_column(dts=dts, dt_df=dt_df, arg=self.monthly, name="month", base=12, add_cosines=self.add_cosines)
        dt_df = DatetimeConversion.add_converted_dt_column(dts=dts, dt_df=dt_df, arg=self.daily, name="day", base=31, add_cosines=self.add_cosines)
        dt_df = DatetimeConversion.add_converted_dt_column(dts=dts, dt_df=dt_df, arg=self.hourly, name="hour", base=24, add_cosines=self.add_cosines)

        idx = list(df.columns).index(self.dt_column)

        for i, c in enumerate(dt_df.columns):
            v = dt_df[c].values

            df.insert(loc=idx+i, column=c, value=v)

        return df

    @staticmethod
    def convert_to_sines(values, base):
        assert isinstance(values, list)
        assert isinstance(base, int)
        return [np.sin(2*np.pi*(x)/base) for x in values]

    @staticmethod
    def convert_to_cosines(values, base):
        assert isinstance(values, list)
        assert isinstance(base, int)
        return [np.cos(2*np.pi*(x)/base) for x in values]
    
    @staticmethod
    def convert_dts_to_ints(dts, attribute):
        return [getattr(dt, attribute) for dt in dts]

    @staticmethod
    def add_converted_dt_column(dts, dt_df, arg, name, base, add_cosines):
        if arg is False:
            pass

        elif arg == "int":
            dt_df[f"{name}"] = DatetimeConversion.convert_dts_to_ints(dts, name)

        elif arg == "sines":
            values = DatetimeConversion.convert_dts_to_ints(dts, name)

            dt_df[f"{name}_sines"] = DatetimeConversion.convert_to_sines(values=values, base=base)

            if add_cosines:
                dt_df[f"{name}_cosines"] = DatetimeConversion.convert_to_cosines(values=values, base=base)

        else:
            raise NotImplementedError

        return dt_df


class AddWeekends(Transform):
    def __init__(self):
        pass

    def __call__(self, df):
        weekends = [int(self.is_weekend(dt)) for dt in df["datetimes"]]

        df["is_weekend"] = weekends

        return df

    @staticmethod
    def is_weekend(dt):
        # 0 through 4 is monday through friday; 5 and 6 indicate saturday and sunday
        return dt.weekday() >= 5


class AddHolidays(Transform):
    def __init__(self, country="DE"):
        self._holidays = holidays.country_holidays(country)

    def is_holiday(self, dt):
        return dt in self._holidays

    def __call__(self, df):
        is_holidays = [int(self.is_holiday(dt)) for dt in df["datetimes"]]
        df["is_holiday"] = is_holidays
        return df

class DropLoadProfileColumns(Transform):
    def __init__(self):
        pass

    def __call__(self, df):
        for c in df.columns:
            if c.startswith("load_profile"):
                df = df.drop(columns=[c])

        return df

class DuplicateColumns(Transform):
    """ Used to duplicate columns; prepends a prefix to the columns name
    """
    def __init__(self, prefixes, exclude=[]):
        self.prefixes = prefixes
        self.exclude = exclude

    def __call__(self, df):
        assert all(excl in df.columns for excl in self.exclude)

        new_df = df.iloc[:, :0]

        for c in df.columns:
            if c in self.exclude:
                new_df[c] = df[c].copy(deep=True)
            else:
                for prefix in self.prefixes:
                    new_df[f"{prefix}_{c}"] = df[c].copy(deep=True)

        return new_df

class AddYesterdaysValue(Transform):
    def __init__(self, h, column="y", convolve=True):
        self.h = h
        self.column = column
        
        self.convolve = convolve

    def __call__(self, df):
        td = get_timedelta(df)

        n_rows_in_horizon = int(self.h*3600 / td)
        n_rows_per_day = int(24*3600 / td)
        n_rows_offset = n_rows_per_day - n_rows_in_horizon

        target = list(df[self.column].values)[:-n_rows_offset]

        if self.convolve:
            weights = [2] + [3]*(len(target)-2) + [2]
            target = np.convolve(target, [1,1,1], "same")/np.array(weights)
            target = list(target)

        values = [np.nan]*n_rows_offset + target


        df[f"yesterdays_{self.column}"] = values

        return df

class AddLastWeeksValue(Transform):
    def __init__(self, h, column="y", convolve=True):
        self.h = h
        self.column = column
        self.convolve = convolve

    def __call__(self, df):
        td = get_timedelta(df)

        n_rows_in_horizon = int(self.h*3600 / td)
        n_rows_per_week = int(7*24*3600 / td)
        n_rows_offset = n_rows_per_week - n_rows_in_horizon

        target = list(df[self.column].values)[:-n_rows_offset]

        if self.convolve:
            weights = [2] + [3]*(len(target)-2) + [2]
            target = np.convolve(target, [1,1,1], "same")/np.array(weights)
            target = list(target)

        values = [np.nan]*n_rows_offset + target


        df[f"last_weeks_{self.column}"] = values

        return df

class DropNaNs(Transform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, df):
        return df.dropna()

class OnlyKeepSpecificColumns(Transform):
    def __init__(self, columns, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(columns, list):
            assert isinstance(columns, str)
            columns = [columns]
        
        self.columns = columns

    def __call__(self, df):
        return df[self.columns]


def get_timedelta(df):
    td = (df["datetimes"].iloc[1] - df["datetimes"].iloc[0]).total_seconds()
    return td  

def get_timedeltas(df):
    dts = df["datetimes"]
    tds = [(dt1 - dt0).total_seconds() for dt1, dt0 in zip(dts[1:], dts[:-1])]
    return tds

def load_df(path):
    df = pd.read_csv(path)
    df._source_path = path
    
    df = df.iloc[TINY_TEST_BEGIN:TINY_TEST_END] if TINY_TEST else df

if __name__ == "__main__":
    from pathlib import Path

    usable_data_folder = Path(r"C:\Users\Flin\OneDrive - TU Eindhoven\Flin\Flin\01 - Uni\00_Internship\Nokia\data_preperation\data\usable")
    fn = r"industrial\LoadProfile_20IPs_2016_LG_1.csv"
    df = pd.read_csv(usable_data_folder / fn, index_col="datetime")
    df.index = pd.to_datetime(df.index)
    
    df["dts"] = [pd.to_datetime(dt) for dt in list(df.index)]

    print(df.head())

    tf1= TimeseriesToRow(df=df, column_name="load_profile", history_window=5)
    tf2 = DatetimeConversion(df=df, dt_column="dts")
    pipe = DataPipeline([tf1, tf2])

    out =pipe(df.copy(deep=True))
    print(out.head())
    
    print("Done!")
