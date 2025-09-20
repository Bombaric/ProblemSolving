#!/usr/bin/env python3
import numpy as np
import pandas as pd
#lavoro con dataframe pandas
#calcolo rendimenti logaritmici e winsorizzazione

def returns(df_prices, method="log", standardize=True):
    #per completezza considero anche i rendimenti semplici, ma di default uso quelli logaritmici
    if method == "log":
        ret = np.log(df_prices / df_prices.shift(1))
    elif method == "simple":
        ret = df_prices.pct_change()
    else:
        raise ValueError("Unknown method: choose 'log' or 'simple'.")
    #normalizzo i rendimenti (z-score) per ogni colonna (ticker). ogni titolo ha media 0 e varianza 1 nel corso del periodo
    if standardize:
        ret = (ret - ret.mean()) / ret.std()
    return ret

#winsorizzazione: taglio delle code della distribuzione per evitare outlier estremi
def winsorize_sigma(df, k=5.0): #windsotizzazione a 5 sigma
    mu = df.mean()
    sd = df.std(ddof=0)
    lo = mu - k * sd
    hi = mu + k * sd
    return df.clip(lower=lo, upper=hi, axis="columns")

