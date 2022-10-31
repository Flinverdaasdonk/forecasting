from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import statsmodels.api as sm
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time

"""
residential_with_pv:
1/72: p=0, q=0, P=0, D=0, Q=0, aic=19544.17, , bic=19549.37, dur=0.04s
2/72: p=0, q=0, P=0, D=0, Q=1, aic=19545.50, , bic=19555.90, dur=31.57s
3/72: p=0, q=0, P=0, D=1, Q=0, aic=18949.83, , bic=18954.96, dur=13.25s
4/72: p=0, q=0, P=0, D=1, Q=1, aic=18338.03, , bic=18348.28, dur=272.26s
5/72: p=0, q=0, P=1, D=0, Q=0, aic=19545.43, , bic=19555.84, dur=11.03s
6/72: p=0, q=0, P=1, D=0, Q=1, aic=19541.13, , bic=19556.74, dur=59.39s
7/72: p=0, q=0, P=1, D=1, Q=0, aic=18576.30, , bic=18586.56, dur=143.98s
8/72: p=0, q=0, P=1, D=1, Q=1, aic=18338.19, , bic=18353.57, dur=509.26s
9/72: p=0, q=1, P=0, D=0, Q=0, aic=19512.06, , bic=19522.46, dur=0.87s
10/72: p=0, q=1, P=0, D=0, Q=1, aic=19512.28, , bic=19527.88, dur=19.26s
11/72: p=0, q=1, P=0, D=1, Q=0, aic=18879.87, , bic=18890.13, dur=28.08s
12/72: p=0, q=1, P=0, D=1, Q=1, aic=18278.88, , bic=18294.26, dur=271.96s
13/72: p=0, q=1, P=1, D=0, Q=0, aic=19512.06, , bic=19527.67, dur=13.17s
14/72: p=0, q=1, P=1, D=0, Q=1, aic=19499.97, , bic=19520.78, dur=100.82s
15/72: p=0, q=1, P=1, D=1, Q=0, aic=18515.64, , bic=18531.02, dur=129.68s
16/72: p=0, q=1, P=1, D=1, Q=1, aic=18278.76, , bic=18299.28, dur=288.58s
17/72: p=0, q=2, P=0, D=0, Q=0, aic=19484.36, , bic=19499.97, dur=1.45s
18/72: p=0, q=2, P=0, D=0, Q=1, aic=19482.70, , bic=19503.51, dur=17.04s
19/72: p=0, q=2, P=0, D=1, Q=0, aic=18817.68, , bic=18833.06, dur=30.92s
20/72: p=0, q=2, P=0, D=1, Q=1, aic=18217.69, , bic=18238.20, dur=298.34s
21/72: p=0, q=2, P=1, D=0, Q=0, aic=19481.50, , bic=19502.31, dur=21.26s
22/72: p=0, q=2, P=1, D=0, Q=1, aic=19460.05, , bic=19486.06, dur=69.69s
23/72: p=0, q=2, P=1, D=1, Q=0, aic=18460.17, , bic=18480.69, dur=245.29s
24/72: p=0, q=2, P=1, D=1, Q=1, aic=18219.52, , bic=18245.16, dur=540.79s
25/72: p=1, q=0, P=0, D=0, Q=0, aic=19522.30, , bic=19532.71, dur=0.56s
26/72: p=1, q=0, P=0, D=0, Q=1, aic=19523.18, , bic=19538.79, dur=13.53s
27/72: p=1, q=0, P=0, D=1, Q=0, aic=18910.21, , bic=18920.46, dur=9.06s
ERROR vanaf hier
"""

def lazy_load_data(path, n_weeks):
    df = pd.read_csv(path)
    
    df = df.iloc[4*24*7:4*24*7*(1+n_weeks)]

    y = df["y"].values

    X = df.drop(columns=["datetimes", "y"]).to_numpy()

    return y, X

def custom_auto_arima(path):
    N_WEEKS = 2
    INCLUDE_X = False
    y, X = lazy_load_data(path, n_weeks=N_WEEKS)

    ps = list(range(0, 3)) # 1
    qs = list(range(0, 3)) # 1
    
    Ps = list(range(0, 1)) # 0
    Ds = list(range(1, 2)) # 1
    Qs = list(range(0, 1)) # 0

    n_its = len(ps)*len(qs)*len(Ps)
    i = 0
    all_results = []
    for p in ps:
        for q in qs:
            for P in Ps:
                for D in Ds:
                    for Q in Qs:
                        tic = time.time()

                        if INCLUDE_X:
                            model =  SARIMAX(endog=y, exog=X, order=(p,1,q), seasonal_order=(P,D,Q, 4*24))
                        else:
                            model =  SARIMAX(endog=y, order=(p,1,q), seasonal_order=(P,D,Q, 4*24))

                        out = model.fit(disp=False)

                        duration = time.time() - tic
                        aic = out.aic
                        bic = out.bic

                        result = [p, q, P, D, Q, duration, aic, bic]
                        all_results.append(result)

                        i += 1
                        print(f"{i}/{n_its}: p={p}, q={q}, P={P}, D={D}, Q={Q}, aic={aic:.2f}, , bic={bic:.2f}, dur={duration:.2f}s")

    df = pd.DataFrame(all_results, columns=["p", "q", "P", "D", "Q", "duration", "aic", "bic"])
    df.to_csv(f"SARIMAX_RESULTS_{path.stem}.csv")





def lazy_auto_arima(path):
    y, X = lazy_load_data(path)

    model = auto_arima(y=y,
                        X=X, 
                        seasonal=True,
                        m=4*24,
                        start_p=0,
                        d=1, # See determine how stationary
                        start_q=0,
                        start_D=0,
                        max_D=1,
                        max_p=4,
                        max_q=4,
                        start_P=0,
                        start_Q=0,
                        max_P=2,
                        max_Q=2,
                        max_order=None,
                        information_criterion="aic",
                        trace=True,
                        error_action="ignore",
                        stepwise=True, 
                        maxiter=10
                        )

    return model



if __name__ == "__main__":
    usable_data_folder = Path(r"C:\Users\Flin\OneDrive - TU Eindhoven\Flin\Flin\01 - Uni\00_Internship\Nokia\00_Programming\forecasting\datasets\train")

    fns = {"residential_with_pv": r"residential_with_pv\h=2_residential_2018_WITH_PV_SFH13_2018.csv",
           "residential_no_pv": r"residential_no_pv\h=2_residential_2018_NO_PV_SFH18_2018.csv",
           "industrial": r"industrial\h=2_industrial_2016_LG_9.csv"}

    models = {}
    print("\n")
    for k, fn in fns.items():
        print(f"{k}:")
        path = usable_data_folder / fn

        custom_auto_arima(path)