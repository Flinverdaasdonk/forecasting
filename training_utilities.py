from pathlib import Path
from config import *


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

if __name__ == "__main__":

    for fn in yield_test_fns(h=2):
        print(Path(fn.parent.parent.stem)/  Path(fn.parent.stem)/  fn.stem)