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
        M = len(g['true_edges'])
        p = sum(g['true_edges'])
        n = M - p
        positiv.append(p)
        negativ.append(n)

        M = len(g['true_nodes'])
        nodes = np.array(g['true_nodes'])
        pp = nodes.sum(axis=0)
        if type(pp) is np.float32:
            continue
        nn = M - pp

        positiv_nodes.append(pp)
        negativ_nodes.append(nn)

    return positiv, negativ, positiv_nodes, negativ_nodes

def calculate_imbalance(dataset):
    positiv, negativ, positiv_nodes, negativ_nodes = _calculate_positiv_negativ(dataset)
    P = np.mean(positiv)
    N = np.mean(negativ)
    PP = np.mean(positiv_nodes, axis=0)
    NN = np.mean(negativ_nodes, axis=0)
    disb_nodes = NN / PP
    disb_nodes[PP == 0.0] = 0
    publaynet_imbalance = (disb_nodes / sum(disb_nodes)).tolist()
    edge_imbalance = float(N / P)

    return publaynet_imbalance, edge_imbalance
