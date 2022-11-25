from pathlib import Path
from config import *
import forecasting_models as models

def generic_yield_fns(h, specific_folder_name, main_data_dir=MAIN_DATA_DIR, time_compression=False, use_aggregate=False):
    assert not (time_compression and use_aggregate), f"tc={time_compression}, aggr={use_aggregate}"

    specific_data_dir = Path(main_data_dir) / specific_folder_name

    if use_aggregate:
        data_categories = ["aggregate"]
    else:
        data_categories = DATA_CATEGORIES

    for data_category in data_categories:
        sub_data_dir = specific_data_dir / data_category
        fns = [fn for fn in sub_data_dir.iterdir() if fn.is_file() and fn.stem.startswith(f"h={h}_")]

        if time_compression:
            fns = [fn for fn in fns if "_TC=" in fn.stem]
        else:
            fns = [fn for fn in fns if "_TC=" not in fn.stem]

        for fn in fns:
            yield fn


def yield_train_fns(h, main_data_dir, time_compression, use_aggregate):
    for fn in generic_yield_fns(h, specific_folder_name="train", main_data_dir=main_data_dir, time_compression=time_compression, use_aggregate=use_aggregate):
        yield fn


def yield_test_fns(h, main_data_dir, time_compression, use_aggregate):
    for fn in generic_yield_fns(h, specific_folder_name="test", main_data_dir=main_data_dir, time_compression=time_compression, use_aggregate=use_aggregate):
        yield fn


def yield_validation_fns(h, main_data_dir, time_compression, use_aggregate):
    for fn in generic_yield_fns(h, specific_folder_name="validation", main_data_dir=main_data_dir, time_compression=time_compression, use_aggregate=use_aggregate):
        yield fn


def yield_all_fns(h, main_data_dir, time_compression, use_aggregate):
    fns = [fn for fn in yield_train_fns(h, main_data_dir, time_compression, use_aggregate)]
    fns.extend([fn for fn in yield_test_fns(h, main_data_dir, time_compression, use_aggregate)])
    fns.extend([fn for fn in yield_validation_fns(h, main_data_dir, time_compression, use_aggregate)])

    for fn in fns:
        yield fn


def load_model(model_type, df, h, adt, data_path, **kwargs):
    
    # ### INITIALIZE MODEL
    if model_type == "RandomForest":
        m = models.CustomRandomForest(df=df, h=h, additional_df_transformations=adt, data_path=data_path, **kwargs)
    elif model_type == "SARIMAX":
        m = models.CustomSARIMAX(df=df, h=h, additional_df_transformations=adt, data_path=data_path, **kwargs)
    elif model_type == "Prophet":
        m = models.CustomProphet(df=df, h=h, additional_df_transformations=adt, data_path=data_path, **kwargs)
    elif model_type == "LSTM":
        m = models.CustomLSTM(df=df, h=h, additional_df_transformations=adt, data_path=data_path, **kwargs)
    elif model_type == "LastWeeks":
        m = models.CustomNaiveLastWeek(df=df, h=h, additional_df_transformations=adt, data_path=data_path, **kwargs)
    elif model_type == "Yesterdays":
        m = models.CustomNaiveYesterday(df=df, h=h, additional_df_transformations=adt, data_path=data_path, **kwargs)

    else:
        raise NotImplementedError

    return m


if __name__ == "__main__":

    for fn in yield_test_fns(h=2):
        print(Path(fn.parent.parent.stem)/  Path(fn.parent.stem)/  fn.stem)