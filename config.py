import sys
import os


WINDOWS_DATA_DIR = r"C:\Users\Flin\OneDrive - TU Eindhoven\Flin\Flin\01 - Uni\00_Internship\Nokia\00_Programming\forecasting\datasets"
WINDOWS_MAIN_LOG_DIR = r"C:\Users\Flin\OneDrive - TU Eindhoven\Flin\Flin\01 - Uni\00_Internship\Nokia\00_Programming\forecasting\logs"

LINUX_DATA_DIR = r""
LINUX_MAIN_LOG_DIR = r""


if 'win' in sys.platform:
    MAIN_DATA_DIR = WINDOWS_DATA_DIR
    MAIN_LOG_DIR = WINDOWS_MAIN_LOG_DIR
    N_CORES = int(os.cpu_count()/2)

elif "linux" in sys.platform:
    MAIN_DATA_DIR = LINUX_DATA_DIR
    MAIN_LOG_DIR = LINUX_MAIN_LOG_DIR
    N_CORES = int(os.cpu_count()/2)

else:
    raise NotImplementedError(sys.platform)


DATA_CATEGORIES = ["industrial", "residential_no_pv", "residential_with_pv"]
TEST_CATEGORY = "industrial"

TINY_TEST = True
TINY_TEST_BEGIN = 4*24*7*1
TINY_TEST_END = 4*24*7*4

ONLY_FIT_USING_LAST_N_WEEKS = 10
ROLLING_PREDICTION = False
TRAIN_EVAL_SPLIT = 0.7
HORIZON = 2
MODEL_TYPE = "RandomForest"  # "RandomForest", "SARIMAX", "Prophet", "RulesBased"