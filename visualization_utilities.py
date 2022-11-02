ROLLING_PREDICTION = True
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

def plot_predictions(ax, model):
    # grab actual y
    y = get_y(model)
    
    # grab yhat
    yhat = get_yhat(model)

    ax.set_title(f"{model.name}: Prediction vs Reality, RP={ROLLING_PREDICTION}")

    x = get_x(model)


    ax.plot(x, y, label="y")
    ax.plot(x, yhat, label="yhat")
    ax.legend()

    return ax



def plot_prediction_error(ax, model):
    # grab actual y
    y = get_y(model)

    # grab yhat
    yhat = get_yhat(model)

    # calculate error
    delta = [_y - _yh for _y, _yh in zip(y, yhat)]
    x = get_x(model)


    ax.set_title(f"{model.name}: Prediction Error,, RP={ROLLING_PREDICTION}")
    ax.plot(x, delta, label="y-yhat")
    ax.legend()

    return ax
