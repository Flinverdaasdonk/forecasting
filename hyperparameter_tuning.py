import optuna
import training as tr
import joblib
import time
from config import *

class IntermediateStudySaver:
    def __init__(self, suffix, save_every_n_calls=2):
        self.suffix = suffix
        self.save_every_n_calls = save_every_n_calls
        self.start = time.time()
        self.called_n_times = 0

    def __call__(self, study, trial):
        self.called_n_times += 1
        now = time.time()
        seconds_passed = self.start - now

        print(f"{self.called_n_times}: minutes passed: {seconds_passed / 60:.2f}m ({seconds_passed / 60 /self.called_n_times:.2f}m per trial)")

        if self.called_n_times % self.save_every_n_calls == 0:
            joblib.dump(study, f"studies/RFR_study_{self.called_n_times}_TT={TINY_TEST}_{self.suffix}.pkl")

        return

def objective(trial): 
    max_features = trial.suggest_float("max_features", 0.2, 1.0)
    max_samples = trial.suggest_float("max_samples", 0.2, 1.0)
    n_estimators = trial.suggest_int("n_estimators", 50, 1000)
    history_depth = trial.suggest_int("history_depth", 0, 6)

    tic = time.time()

    total_mse = tr.hyperparameter_sweep(max_features=max_features, max_samples=max_samples, n_estimators=n_estimators, history_depth=history_depth) # contains 11 files

    total_time = time.time() - tic


    return total_time, total_mse


if __name__ == "__main__":

    intermediate_study_saver_callback = IntermediateStudySaver(suffix="A")
    study = optuna.create_study(study_name="RFR_optimization", directions=["minimize", "minimize"], storage="sqlite:///RFR.db", load_if_exists=True)

    study.optimize(objective, n_trials=500, n_jobs=1, callbacks=[intermediate_study_saver_callback])

    # study = joblib.load("studies/RFR_study.pkl")

    # #optuna.visualization.plot_pareto_front(study, target_names=["total_time", "total_mse"])

    # print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    # trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])
    # print(f"Trial with highest accuracy: ")
    # print(f"\tnumber: {trial_with_highest_accuracy.number}")
    # print(f"\tparams: {trial_with_highest_accuracy.params}")
    # print(f"\tvalues: {trial_with_highest_accuracy.values}")


    # # print("Best trial until now:")
    # # print(" Value: ", study.best_trials.value)
    # # print(" Params: ")
    # # for key, value in study.best_trial.params.items():
    # #     print(f"    {key}: {value}")
