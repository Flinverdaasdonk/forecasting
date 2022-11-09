from config import *
import logging_utilities as lut

def get_y(model):
    y = list((model.transformed_df["y"].values))
    return y

def get_yhat(model):
    yhat_train = list(model.predict(predict_on_test=False))

    lut.blockPrint()
    yhat_test = list(model.predict(predict_on_test=True, rolling_prediction=ROLLING_PREDICTION))
    lut.enablePrint()

    
    yhat = yhat_train + yhat_test
    return yhat

def get_x(model):
    x = model.get_corresponding_dts(df=model.transformed_df)
    return x

