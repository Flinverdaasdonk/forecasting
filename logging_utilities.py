from config import *
from pathlib import Path
import json

def find_next_available_ID(log_dir=MAIN_LOG_DIR):
    files = [file for file in Path(log_dir).iterdir() if file.is_file()]

    if len(files) == 0:
        id = 0

    else:
        ids = [int(file.stem.split("_")[0]) for file in files]
        highest_used_id = max(ids)
        id = highest_used_id + 1

    return id

def generate_fn(model):
    # fn format is: ID_h={HORIZON}_model_name
    id = find_next_available_ID()
    h = model.h
    model_name = model.name

    fn = f"{id}_h={h}_{model_name}"

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

def make_and_save_logs(model, logs=None, log_dir=MAIN_LOG_DIR):
    logs = make_logs(model, logs)
    logs["rolling_prediction"] = ROLLING_PREDICTION
    logs["train_eval_split"] = TRAIN_EVAL_SPLIT
    save_logs(model, logs, log_dir)

    return