#!/usr/bin/env python3
#r statistics of consecutive eigenvalues

def r_statistics(eig_sorted):
    r_stats = []
    for i in range(1, len(eig_sorted)-1):
        delta_n = eig_sorted[i] - eig_sorted[i-1]
        delta_n_plus_1 = eig_sorted[i+1] - eig_sorted[i]
        r = min(delta_n, delta_n_plus_1) / max(delta_n, delta_n_plus_1)
        r_stats.append(r)
        mean_r = sum(r_stats) / len(r_stats)
    return mean_r

def ipr(vecs):
    ipr = []
    for vec in vecs.T:
        ipr_value = sum(vec**4) / (sum(vec**2)**2)
        ipr.append(ipr_value)
    return ipr