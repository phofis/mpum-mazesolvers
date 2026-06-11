import numpy as np
from numpy.typing import NDArray

NEG_INF = -np.inf

def apply_mask(q_values: NDArray, mask: NDArray):
    return np.where(mask, q_values, NEG_INF)
def masked_argmax(q_values: NDArray, mask: NDArray):
    q = apply_mask(q_values, mask)
    max_val = np.max(q)
    best_actions = np.flatnonzero(q == max_val)
    return np.random.choice(best_actions)
def masked_max(q_values: NDArray, mask: NDArray):
    q = apply_mask(q_values, mask)
    return np.max(q)
def masked_softmax_sample(q_values: NDArray, mask: NDArray, temperature: float):
    valid = np.flatnonzero(mask)
    q = q_values[valid]
    q = q / max(temperature, 1e-8)
    exp_q = np.exp(q - np.max(q))
    probs = exp_q / np.sum(exp_q)

    return np.random.choice(valid, p=probs)
def sample_valid(mask: NDArray) -> int:
    valid = np.flatnonzero(mask)
    return np.random.choice(valid)