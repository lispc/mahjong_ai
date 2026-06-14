#!/usr/bin/env python3
"""Tests for ShantenUkeire agent."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import algo.agents.shanten_ukeire as su
import algo.context.v3 as context_v3
import tile


def test_known_hand_selects_fa_cai():
    """尾盘那手牌：打发财优于打四万。"""
    hand14 = [2, 3, 4, 4, 13, 15, 16, 17, 18, 18, 22, 23, 24, 37]
    c = context_v3.ContextV3()
    # 模拟已见牌：让四万、发财都还在牌山
    discards = [
        5, 6, 9,       # 五万,六万,九万
        13,            # 三条
        22, 23, 26, 27, 28,  # 二筒,三筒,六筒,七筒,八筒
        32, 33, 34, 35, 35,  # 西风,南风,北风,红中,红中
    ]
    for t in discards:
        c.see_tile(t, 'self')

    disc = su.select(hand14, c)
    assert disc == 37, f"expected 发财 (37), got {tile.tile_to_str(disc)} ({disc})"


def test_leaf_value_tenpai_vs_far():
    """听牌手 leaf 价值应远高于 3 向听手。"""
    c = context_v3.ContextV3()
    tenpai = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16]
    far = [1, 2, 4, 5, 7, 8, 11, 12, 14, 15, 21, 22, 24]
    assert su.leaf_value(tenpai, c) > su.leaf_value(far, c)


def test_suv3_selects_fa_cai_with_defense():
    """SUv3 在 defense_weight=2 下应避开点炮四万。"""
    from algo.agents.shanten_ukeire import ShantenUkeireV3Agent
    hand14 = [2, 3, 4, 4, 13, 15, 16, 17, 18, 18, 22, 23, 24, 37]
    c = context_v3.ContextV3()
    discards = [
        5, 6, 9, 13, 22, 23, 26, 27, 28, 32, 33, 34, 35, 35,
    ]
    for t in discards:
        c.see_tile(t, 'self')

    agent = ShantenUkeireV3Agent('SUv3', defense_weight=2.0)
    agent.context = c
    agent.cur = list(hand14)
    disc = agent.next()
    assert disc != 4, f"expected not 四万 (4), got {tile.tile_to_str(disc)} ({disc})"


if __name__ == '__main__':
    test_known_hand_selects_fa_cai()
    test_leaf_value_tenpai_vs_far()
    test_suv3_selects_fa_cai_with_defense()
    print('All ShantenUkeire tests passed.')
