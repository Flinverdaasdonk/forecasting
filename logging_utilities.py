from config import *
from pathlib import Path
import json
import evaluation_utilities as eut
import time
import sys
import os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



def find_next_available_ID(log_dir=MAIN_LOG_DIR):
    files = [file for file in Path(log_dir).iterdir() if file.is_file()]

    if len(files) == 0:
        id = 0

    else:
        ids = [int(file.stem.split("_")[0]) for file in files]
        highest_used_id = max(ids)
        id = highest_used_id + 1

    return id

def generate_fn(model, prefix=LOG_NAME_PREFIX):
    # fn format is: ID_h={HORIZON}_model_name
    id = find_next_available_ID()
    h = model.h
    model_name = model.name

    if LOG_NAME_PREFIX is None or len(LOG_NAME_PREFIX) == 0:
        fn = f"{id}_h={h}_{model_name}"
    else:
        fn = f"{id}_{LOG_NAME_PREFIX}_h={h}_{model_name}"

    return fn

def make_logs(model, logs=None):
    if logs is None:
        logs = {}

    logs = {**logs, **model.logworthy_attributes()}

    return logs

def save_logs(model, logs, log_dir=MAIN_LOG_DIR):
    log_fn = Path(log_dir) / generate_fn(model)

    with open(f"{log_fn}.json", "w") as f:
        json.dump(logs, f)

    return

def make_and_save_logs(model, logs=None, get_x=True, get_y=True, get_yhat=True, log_dir=MAIN_LOG_DIR):
    if logs is None:
        logs = {}

    if get_x:
        x = eut.get_x(model)
        x = [str(_x) for _x in x]
        logs["x"] = x
    if get_y:
        y = eut.get_y(model)
        y = [float(_y) for _y in y]
        logs["y"] = y

    if get_yhat:

        tic = time.time()
        
        yhat = eut.get_yhat(model)
        
        elapsed = time.time() - tic
        logs["time_to_predict"] = elapsed

        yhat = [float(_yh) for _yh in yhat]
        logs["yhat"] = yhat
        
    
    logs = make_logs(model, logs)
    logs["rolling_prediction"] = ROLLING_PREDICTION

    logs["tiny_test"] = TINY_TEST
    if TINY_TEST:
        logs["tiny_test_begin"] = TINY_TEST_BEGIN
        logs["tiny_test_end"]  = TINY_TEST_END

    logs["os"] = str(sys.platform)
    logs["time_at_saving"] = repr(time.ctime())

    save_logs(model, logs, log_dir)

    return