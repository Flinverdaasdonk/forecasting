MAIN_DATA_DIR = r"C:\Users\Flin\OneDrive - TU Eindhoven\Flin\Flin\01 - Uni\00_Internship\Nokia\00_Programming\forecasting\datasets"
MAIN_LOG_DIR = r"C:\Users\Flin\OneDrive - TU Eindhoven\Flin\Flin\01 - Uni\00_Internship\Nokia\00_Programming\forecasting\logs"

DATA_CATEGORIES = ["industrial", "residential_no_pv", "residential_with_pv"]
TEST_CATEGORY = "residential_no_pv"

TINY_TEST = False
TINY_TEST_BEGIN = 4*24*7*2
TINY_TEST_END = 4*24*7*4

ROLLING_PREDICTION = False
TRAIN_EVAL_SPLIT = 0.7
HORIZON = 2
MODEL_TYPE = "RulesBased"  # "RandomForest", "SARIMAX", "Prophet", "RulesBased"