import itertools
import logging
import random

_log = logging.getLogger(__name__)


def product(**axes):
    """Generate a Cartesian product over parameter axes.

    Yields:
        tuple of (key, value) pairs — one combination.
    """
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    for combo in itertools.product(*values):
        yield tuple(zip(keys, combo))


def random_sample(seed: int, n_configs: int, allow_repeats: bool = False, **axes):
    """Generate *n_configs* random configurations from the parameter space.

    Args:
        seed: Seed for the random number generator (reproducibility).
        n_configs: Number of configurations to draw.
        allow_repeats: If False (default), deduplicate and only return
            unique configs.  If True, duplicates may occur.
        **axes: Parameter space axes.  Values are either:
            - ``list`` — categorical / discrete choices, or
            - ``tuple`` of ``(low, high)`` — continuous range sampled
              uniformly and rounded to 2 decimal places.

    Returns:
        List of dicts, each representing one random configuration.
        When *allow_repeats* is False and the total number of unique
        combinations is smaller than *n_configs*, all unique
        combinations are returned and a warning is issued.
    """
    rng = random.Random(seed)
    keys = list(axes.keys())
    values = list(axes.values())

    def _draw():
        cfg = {}
        for k, v in zip(keys, values):
            if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                low, high = v
                cfg[k] = round(rng.uniform(low, high), 2)
            else:  # categorical / discrete
                cfg[k] = rng.choice(v)
        return cfg

    # Count total unique combinations (exact for discrete, estimate for continuous)
    total_unique = 1
    for v in values:
        if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            total_unique = float("inf")  # continuous → unbounded in practice
        else:
            total_unique *= len(v)

    if allow_repeats:
        return [_draw() for _ in range(n_configs)]

    if not isinstance(total_unique, float) and n_configs > total_unique:
        _log.warning(
            "Requested %d unique configs but only %d possible. Returning all unique combos.",
            n_configs, total_unique,
        )
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    # Deduplication via hash of frozenset
    configs = []
    seen = set()

    max_attempts = n_configs * 1000
    attempts = 0

    while len(configs) < n_configs and attempts < max_attempts:
        cfg = _draw()
        cfg_hash = frozenset(cfg.items())
        if cfg_hash not in seen:
            seen.add(cfg_hash)
            configs.append(cfg)
        attempts += 1

    if len(configs) < n_configs:
        _log.warning(
            "Only %d unique configs generated after %d attempts (requested %d).",
            len(configs), attempts, n_configs,
        )

    return configs
