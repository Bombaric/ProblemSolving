#!/usr/bin/env python3
import numpy as np

def returns(df_prices, method="log", standardize=True):
    #per completezza considero anche i rendimenti semplici, ma di default uso quelli logaritmici
    if method == "log":
        ret = np.log(df_prices / df_prices.shift(1))
    elif method == "simple":
        ret = df_prices.pct_change()
    else:
        raise ValueError("Unknown method: choose 'log' or 'simple'.")

    if standardize:
        ret = (ret - ret.mean()) / ret.std()
    return ret

def volatility(df_prices, window=30):
    log_returns = returns(df_prices, method="log", standardize=False)
    vol = log_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility
    return vol

def winsorize_sigma(df, k=5.0): #windsotizzazione a 5 sigma (opzionale)
    mu = df.mean()
    sd = df.std(ddof=0)
    lo = mu - k * sd
    hi = mu + k * sd
    return df.clip(lower=lo, upper=hi, axis="columns")

