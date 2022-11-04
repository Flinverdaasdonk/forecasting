from config import *

def find_next_available_ID(log_dir=MAIN_LOG_DIR):
    files = [file for file in log_dir.iterdir() if file.is_file()]

    if len(files) == 0:
        id = 0

    else:
        ids = [int(file.split("_")[0]) for file in files]
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