import sys
import os

WINDOWS_DATA_DIR = r"C:\Users\Flin\OneDrive - TU Eindhoven\Flin\Flin\01 - Uni\00_Internship\Nokia\00_Programming\forecasting\datasets"
WINDOWS_MAIN_LOG_DIR = r"C:\Users\Flin\OneDrive - TU Eindhoven\Flin\Flin\01 - Uni\00_Internship\Nokia\00_Programming\forecasting\logs"

LINUX_DATA_DIR = r"datasets"
LINUX_MAIN_LOG_DIR = r"logs"

if 'win' in sys.platform:
    MAIN_DATA_DIR = WINDOWS_DATA_DIR
    MAIN_LOG_DIR = WINDOWS_MAIN_LOG_DIR
    N_CORES = 6
    EVAL_TYPE = "test"

elif "linux" in sys.platform:
    MAIN_DATA_DIR = LINUX_DATA_DIR
    MAIN_LOG_DIR = LINUX_MAIN_LOG_DIR
    N_CORES = 38 # int(os.cpu_count() - 8)  
    EVAL_TYPE = "test"

else:
    raise NotImplementedError(sys.platform)


DATA_CATEGORIES = ["industrial", "residential_no_pv", "residential_with_pv"]
TEST_CATEGORY = "industrial"

LOG_NAME_PREFIX = "HPTED"

TINY_TEST = False
TINY_TEST_BEGIN = 4*24*7*1
TINY_TEST_END = 4*24*7*3

if TINY_TEST:
    HORIZONS = [2, 6]
else:
    HORIZONS = [1, 2, 4, 6, 12, 18]

ONLY_FIT_USING_LAST_N_WEEKS = 20
PROPHET_ONLY_FIT_USING_LAST_N_WEEKS = 10
SARIMAX_ONLY_FIT_USING_LAST_N_WEEKS = 2

ROLLING_PREDICT_DAYS_TO_REFIT = 7
ROLLING_PREDICTION = True

TRAIN_EVAL_SPLIT = 0.7

HORIZON = 2

MODEL_TYPE = "LSTM"  # "RandomForest", "SARIMAX", "Prophet", "LastWeeks"