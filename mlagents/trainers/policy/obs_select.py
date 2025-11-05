from typing import List, Tuple

def pick_state_obs_index(obs_shapes: List[Tuple[int, ...]]) -> int:
    """
    Prefer a 1D vector of length exactly 5 (per-agent state token).
    If multiple exist, take the last one. If none exist, return -1.
    """
    candidates = [i for i, s in enumerate(obs_shapes) if len(s) == 1 and s[0] == 5]
    return candidates[-1] if candidates else -1
