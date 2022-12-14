import matplotlib.pyplot as plt
import forecasting_models as models
import evaluation_utilities as eut
from config import *

def calc_features(model, x=True, y=True, yhat=True, delta=True):
    results = {}

    # grab x
    results["x"] = eut.get_x(model) if x else None

    # grab actual y
    results["y"] = eut.get_y(model) if y else None
    
    # grab yhat

    results["yhat"] = eut.get_yhat(model) if yhat else None

    # delta
    if delta:
        if y and yhat:
            d = [_y - _yh for _y, _yh in zip(results["y"], results["yhat"])]
        else:
            print(f"Can't calculate delta, y={y}, yhat={yhat}")
            d = None

        results["delta"] = d


    return results

def basic_plot(model):
    fig, axs = plt.subplots(2, figsize=(8, 8))

    results = calc_features(model)

    plot_predictions(ax=axs[0], results=results,  model=model)
    plot_prediction_error(ax=axs[1], results=results, model=model)

    plt.show()


def plot_predictions(ax, results, model):


    ax.set_title(f"{model.name}: Prediction vs Reality, RP={ROLLING_PREDICTION}")

    
    x = results["x"]
    y = results["y"]
    yhat = results["yhat"]

    ax.plot(x, y, label="y")
    ax.plot(x, yhat, label="yhat")
    ax.legend()

    return ax

def plot_prediction_error(ax, results, model):

    x = results["x"]
    delta = results["delta"]

    ax.set_title(f"{model.name}: Prediction Error, RP={ROLLING_PREDICTION}")
    ax.plot(x, delta, label="y-yhat")
    ax.legend()

    return ax

def plot_feature_importance(m):
    if isinstance(m, models.CustomRandomForest):

        # plot feature importance
        features = m.features

        importances = m.model.feature_importances_

        assert len(features) == len(importances)
        d = {f:i for f, i in zip(features, importances)}

        plt.barh(features, importances)
        plt.show()


        print("Done!")
