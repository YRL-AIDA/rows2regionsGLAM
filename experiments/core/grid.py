import itertools


def product(**axes):
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    for combo in itertools.product(*values):
        yield tuple(zip(keys, combo))
