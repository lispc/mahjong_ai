# -*- coding: utf-8 -*-
"""晋北麻将（推倒胡）JAX 函数式对局环境。

用法：
    import jax
    from jaxenv import env
    state = env.init(jax.random.PRNGKey(0))
    mask = env.legal_mask(state)
    state, reward, done = env.step(state, action)
"""

from . import env, rules
from .env import (
    State, init, step, legal_mask,
    N_ACTIONS,
    A_PASS, A_PENG, A_GANG, A_HU, A_TENPAI_YES, A_TENPAI_NO,
    PHASE_DISCARD, PHASE_CLAIM, PHASE_TENPAI,
    WIN_NONE, WIN_SELF, WIN_RON, WIN_DRAW,
    REWARD_SCORE, REWARD_WINLOSS, DEFAULT_REWARD_KIND,
)

__all__ = [
    'env', 'rules', 'State', 'init', 'step', 'legal_mask', 'N_ACTIONS',
    'A_PASS', 'A_PENG', 'A_GANG', 'A_HU', 'A_TENPAI_YES', 'A_TENPAI_NO',
    'PHASE_DISCARD', 'PHASE_CLAIM', 'PHASE_TENPAI',
    'WIN_NONE', 'WIN_SELF', 'WIN_RON', 'WIN_DRAW',
    'REWARD_SCORE', 'REWARD_WINLOSS', 'DEFAULT_REWARD_KIND',
]
