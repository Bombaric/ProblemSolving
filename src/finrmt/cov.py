#!/usr/bin/env python3
import pandas as pd
def pearson(df):
    return df.corr(method='pearson')

def kendall(df):
    return df.corr(method='kendall')

def mp_clip(R, return_df=True, tickers=None, eps=1e-12):
    """
    MP clipping su matrice di correlazione stimata da R (returns standardizzati).
    R: array T×N o DataFrame (righe=time, colonne=ticker), returns già standardizzati (z-score).
    return_df: se True, restituisce DataFrame con indici/colonne = tickers o R.columns.
    """
    # --- prepara R e nomi colonne ---
    if isinstance(R, pd.DataFrame):
        tickers = R.columns if tickers is None else tickers
        R = R.to_numpy(dtype=float)
    elif tickers is None:
        tickers = [f"x{i}" for i in range(R.shape[1])]

    T, N = R.shape
    q = N / T

    # --- correlazione grezza (da R standardizzato) ---
    C = (R.T @ R) / T

    # --- eigendecomp simmetrica ---
    evals, evecs = np.linalg.eigh(C)

    # --- limiti MP (σ²=1 perché R è standardizzato) ---
    lam_minus = (1 - np.sqrt(q))**2
    lam_plus  = (1 + np.sqrt(q))**2

    # --- mask bulk e media del bulk ---
    bulk_mask = (evals >= lam_minus) & (evals <= lam_plus)
    if bulk_mask.any():
        mean_bulk = evals[bulk_mask].mean()
        evals_clipped = np.where(bulk_mask, mean_bulk, evals)
    else:
        # niente bulk → non faccio clipping
        evals_clipped = evals.copy()

    # --- evita negatività numerica ---
    evals_clipped = np.clip(evals_clipped, eps, None)

    # --- ricostruzione e simmetrizzazione ---
    C_clip = (evecs @ np.diag(evals_clipped) @ evecs.T)
    C_clip = 0.5 * (C_clip + C_clip.T)

    # --- riportiamo a correlazione (diag=1) ---
    d = np.sqrt(np.clip(np.diag(C_clip), eps, None))
    Dinv = np.diag(1.0 / d)
    C_clip = Dinv @ C_clip @ Dinv
    np.fill_diagonal(C_clip, 1.0)

    if return_df:
        return pd.DataFrame(C_clip, index=tickers, columns=tickers)
    return C_clip

def ledoit_wolf(df):
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf()
    lw.fit(df.dropna())
    return pd.DataFrame(lw.covariance_, index=df.columns, columns=df.columns)

import numpy as np
