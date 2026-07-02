# -*- coding: utf-8 -*-
"""把单局 result 转成每个座位的标量奖励。

engine.play_game 返回的 result:
    {
        'winner': name | None,
        'win_type': 'self' | 'ron' | 'draw',
        'players_order': [name, ...],
        'dealer': name          # 仅 ron 时存在，表示放铳（打出致命牌）的玩家
    }

奖励语义（可配置）：
    win:        自摸 / 荣和         -> +1.0
    deal_in:    被别人荣和（自己点炮）-> deal_in（默认 -1.0，可加大以强化防守）
    other_loss: 别人自摸 / 荣了别家   -> other_loss（默认 -1.0）
    draw:       流局                -> 0.0

默认 deal_in == other_loss == -1.0，与 algo.nn.mc_value._outcome_for 一致，
从而与 warm-start value head 的 ±1 语义兼容。若要做防守 shaping，可把 deal_in
调到更负（如 -1.5）。
"""

DEFAULT_REWARD = {
    'win': 1.0,
    'deal_in': -1.0,
    'other_loss': -1.0,
    'draw': 0.0,
}


def seat_reward(result, seat_name, cfg=None):
    """返回 seat_name 在本局的终局奖励标量。"""
    cfg = cfg or DEFAULT_REWARD
    win_type = result.get('win_type')
    if win_type == 'draw':
        return float(cfg['draw'])
    winner = result.get('winner')
    if winner == seat_name:
        return float(cfg['win'])
    # 别人赢了
    if win_type == 'ron' and result.get('dealer') == seat_name:
        return float(cfg['deal_in'])
    return float(cfg['other_loss'])


def terminal_reason(result, seat_name):
    """给日志/分析用的可读终局类型。"""
    win_type = result.get('win_type')
    if win_type == 'draw':
        return 'draw'
    winner = result.get('winner')
    if winner == seat_name:
        return 'tsumo_win' if win_type == 'self' else 'ron_win'
    if win_type == 'ron' and result.get('dealer') == seat_name:
        return 'deal_in'
    if win_type == 'self':
        return 'lose_tsumo'
    return 'lose_ron_others'
