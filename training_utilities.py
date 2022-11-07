from pathlib import Path
from config import *
import forecasting_models as models

def generic_yield_fns(h, specific_folder_name, main_data_dir=MAIN_DATA_DIR):
    specific_data_dir = Path(main_data_dir) / specific_folder_name

    for data_category in DATA_CATEGORIES:
        sub_data_dir = specific_data_dir / data_category
        fns = [fn for fn in sub_data_dir.iterdir() if fn.is_file() and fn.stem.startswith(f"h={h}_")]

        for fn in fns:
            yield fn


def yield_train_fns(h, main_data_dir=MAIN_DATA_DIR):
    for fn in generic_yield_fns(h, specific_folder_name="train", main_data_dir=main_data_dir):
        yield fn


def yield_test_fns(h, main_data_dir=MAIN_DATA_DIR):
    for fn in generic_yield_fns(h, specific_folder_name="test", main_data_dir=main_data_dir):
        yield fn


def yield_validation_fns(h, main_data_dir=MAIN_DATA_DIR):
    for fn in generic_yield_fns(h, specific_folder_name="validation", main_data_dir=main_data_dir):
        yield fn


def yield_all_fns(h, main_data_dir=MAIN_DATA_DIR):
    fns = [fn for fn in yield_train_fns(h, main_data_dir)]
    fns.extend([fn for fn in yield_test_fns(h, main_data_dir)])
    fns.extend([fn for fn in yield_validation_fns(h, main_data_dir)])

    for fn in fns:
        yield fn


def load_model(df, h, adt, data_path):
    # ### INITIALIZE MODEL
    if MODEL_TYPE == "RandomForest":
        m = models.CustomRandomForest(df=df, h=h, additional_df_transformations=adt, data_path=data_path)
    elif MODEL_TYPE == "SARIMAX":
        m = models.CustomSARIMAX(df=df, h=h, additional_df_transformations=adt, data_path=data_path)
    elif MODEL_TYPE == "Prophet":
        m = models.CustomProphet(df=df, h=h, additional_df_transformations=adt, data_path=data_path)
    elif MODEL_TYPE == "RulesBased":
        m = models.CustomSimpleRulesBased(df=df, h=h, additional_df_transformations=adt, data_path=data_path)

    else:
        raise NotImplementedError

    return m


if __name__ == "__main__":

    for fn in yield_test_fns(h=2):
        print(Path(fn.parent.parent.stem)/  Path(fn.parent.stem)/  fn.stem)