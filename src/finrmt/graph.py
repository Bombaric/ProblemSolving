#!/usr/bin/env python3

#MST, clustering gerarchico

def correlation_to_mst(C, labels):
    """
    C: matrice di correlazione N×N (np.ndarray)   (se hai un DataFrame fai .values)
    labels: lista di etichette per i nodi (len == N), es. list(ret.columns)
    Ritorna: grafo MST (networkx.Graph) con nodi = labels e pesi = distanza
    """
    import numpy as np
    import pandas as pd
    import networkx as nx
    from scipy.sparse.csgraph import minimum_spanning_tree

    # 1) assicurati che C sia numerica e simmetrica nel range [-1,1]
    C = np.asarray(C, dtype=float)
    C = np.clip(C, -1.0, 1.0)
    C = 0.5 * (C + C.T)

    # 2) distanza d_ij = sqrt(2*(1 - rho_ij)) ; diag=0
    D = np.sqrt(2.0 * (1.0 - C))
    np.fill_diagonal(D, 0.0)

    # 3) MST sulla matrice delle distanze
    mst_sparse = minimum_spanning_tree(D)          # restituisce matrice sparsa (diretta)
    mst_dense  = mst_sparse.toarray()
    # Rendi simmetrico (MST non orientato)
    mst_dense  = np.maximum(mst_dense, mst_dense.T)

    # 4) DataFrame di adiacenza con etichette dei nodi (hashable, es. stringhe)
    A = pd.DataFrame(mst_dense, index=labels, columns=labels)

    # 5) Crea grafo non orientato da adiacenza (pesi in 'weight')
    G = nx.from_pandas_adjacency(A, create_using=nx.Graph)
    return G
