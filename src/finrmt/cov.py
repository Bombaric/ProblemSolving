#!/usr/bin/env python3
import pandas as pd
def pearson(df):
    return df.corr(method='pearson')

def kendall(df):
    return df.corr(method='kendall')