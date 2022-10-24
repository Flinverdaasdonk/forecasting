

def plot_predictions(ax, model):
    # grab actual y
    y = list((model.transformed_data["y"].values))

    # grab predictions, and concat
    yhat_train = list(model.predict(data=model.train_data))
    yhat_test = list(model.predict())
    yhat = yhat_train + yhat_test

    ax.set_title(f"{model.name}: Prediction Error")

    x = model.get_corresponding_dts(df=model.transformed_data)

    ax.plot(x,y, label="y")
    ax.plot(x, yhat, label="yhat")
    ax.legend()

    return ax



def plot_prediction_error(ax, model):

    # grab actual y
    y = list((model.transformed_data["y"].values))

    # grab predictions, and concat
    yhat_train = list(model.predict(data=model.train_data))
    yhat_test = list(model.predict())
    yhat = yhat_train + yhat_test

    # calculate error
    delta = [_y - _yh for _y, _yh in zip(y, yhat)]
    x = model.get_corresponding_dts(df=model.transformed_data)


    ax.set_title(f"{model.name}: Prediction Error")
    ax.plot(x, delta, label="y-yhat")
    ax.legend()

    return ax

    # # plot real vs predicted
    # plt.figure()
    # plt.title(crf.name)
    # plt.plot(y, label="y")
    # plt.plot(yhat, label="yhat")
    # plt.legend()
    # plt.show()