import numpy as np

def _calculate_positiv_negativ(dataset):
    positiv = []
    negativ = []
    positiv_nodes = []
    negativ_nodes = []
    k = 0
    for g in dataset:
        if not "true_edges" in g or not "true_nodes" in g:
            continue
        M = len(g['true_edges'].cpu())
        p = sum(g['true_edges'].cpu())
        n = M - p
        positiv.append(p)
        negativ.append(n)

        M = len(g['true_nodes'])
        nodes = np.array(g['true_nodes'].cpu())
        pp = nodes.sum(axis=0)
        nn = M - pp

        positiv_nodes.append(pp)
        negativ_nodes.append(nn)

    return positiv, negativ, positiv_nodes, negativ_nodes

def calculate_imbalance(dataset):
    positiv, negativ, positiv_nodes, negativ_nodes = _calculate_positiv_negativ(dataset)
    P = np.mean(positiv)
    N = np.mean(negativ)
    PP = np.atleast_1d(np.mean(positiv_nodes, axis=0))
    NN = np.atleast_1d(np.mean(negativ_nodes, axis=0))
    disb_nodes = np.where(PP == 0.0, 0.0, NN / PP)
    if disb_nodes.sum() == 0:
        publaynet_imbalance = disb_nodes.tolist()
    else:
        publaynet_imbalance = (disb_nodes / disb_nodes.sum()).tolist()
    edge_imbalance = float(N / P)

    return publaynet_imbalance, edge_imbalance
