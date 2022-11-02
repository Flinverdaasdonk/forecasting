import matplotlib.pyplot as plt

ROLLING_PREDICTION = False

def get_y(model):
    y = list((model.transformed_df["y"].values))
    return y

def get_yhat(model):
    yhat_train = list(model.predict(predict_on_test=False))
    yhat_test = list(model.predict(predict_on_test=True, rolling_prediction=ROLLING_PREDICTION))
    yhat = yhat_train + yhat_test
    return yhat

def get_x(model):
    x = model.get_corresponding_dts(df=model.transformed_df)
    return x

def calc_features(model, x=True, y=True, yhat=True, delta=True):
    results = {}

    # grab x
    results["x"] = get_x(model) if x else None

    # grab actual y
    results["y"] = get_y(model) if y else None
    
    # grab yhat

    results["yhat"] = get_yhat(model) if yhat else None

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
