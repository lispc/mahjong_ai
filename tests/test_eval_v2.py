import sys
sys.path.insert(0, '/Users/zhangzhuo/repos/personal/mahjong_ai')

import algo.eval.v2 as eval_v2
from tile import *


def assert_eq(a, b, msg=""):
    assert a == b, f"{msg}: {a} != {b}"


def test_win():
    # 一般胡牌：4 面子 + 1 对子
    hand = [wan1, wan2, wan3, wan4, wan5, wan6, suo1, suo1, suo1, tong2, tong3, tong4, east, east]
    assert eval_v2.is_win(hand), "一般胡牌应成立"


def test_shanten_basic():
    # 完全散牌 13 张（同花色内至少间隔 3，避免形成任何搭子）
    # 七对子路径决定其向听数为 6
    hand = [wan1, wan4, wan7, suo2, suo5, suo8, tong1, tong4, tong7, east, south, west, north]
    assert_eq(eval_v2.shanten(hand), 6, "完全散牌应向听 6（七对子路径）")

    # 一手已经听牌（一般型）
    hand = [wan1, wan2, wan3, wan4, wan5, wan6, wan7, wan8, wan9, tong1, tong1, suo2, suo2]
    assert_eq(eval_v2.shanten(hand), 0, "该手牌应听牌")
    assert eval_v2.tenpai_tiles(hand) >= 1

    # 5 对 + 3 单：实际上已听牌（可进 1 或 2 完成 4 面子 + 1 对）
    hand = [wan1, wan1, wan2, wan2, wan3, wan3, wan4, wan4, wan5, wan5, suo1, suo2, suo3]
    assert_eq(eval_v2.shanten(hand), 0, "5 对 + 顺子余牌应已听牌")


def test_seven_pairs():
    hand = [wan1, wan1, wan2, wan2, wan3, wan3, wan4, wan4, wan5, wan5, wan6, wan6, wan7]
    assert_eq(eval_v2.shanten(hand), 0, "七对子一向听应向听 0")
    assert eval_v2.tenpai_tiles(hand) >= 1, "七对子一向听应至少待 1 张"

    hand = [wan1, wan1, wan2, wan2, wan3, wan3, wan4, wan4, wan5, wan5, wan6, wan6, wan7, wan7]
    assert eval_v2.is_win(hand), "七对应胡牌"
    assert_eq(eval_v2.shanten(hand), -1, "七对胡牌应向听 -1")


def test_eval_compare():
    # 好牌 vs 差牌
    good = [wan1, wan2, wan3, wan4, wan5, wan6, wan7, wan8, wan9, tong1, tong1, suo2, suo2]
    bad = [wan1, wan4, wan7, suo2, suo5, suo8, tong1, tong4, tong7, east, south, west, north]
    assert eval_v2.evaluate(good) > eval_v2.evaluate(bad), "好牌分数应更高"


def test_taatsu_quality():
    # 两面搭子多应该分高
    taatsu_hand = [wan4, wan5, suo5, suo6, tong6, tong7, wan1, wan9, east, west, south, north, blank]
    print('taatsu quality:', eval_v2.taatsu_quality(taatsu_hand))
    assert eval_v2.taatsu_quality(taatsu_hand) > 5


def test_known_shanten():
    # 1 向听：3 面子 + 1 对子 + 1 单张
    hand = [wan1, wan2, wan3, wan4, wan5, wan6, tong1, tong1, suo2, suo3, suo4, wan7, east]
    assert_eq(eval_v2.shanten(hand), 1, "应为 1 向听")


if __name__ == '__main__':
    test_win()
    test_shanten_basic()
    test_seven_pairs()
    test_eval_compare()
    test_taatsu_quality()
    test_known_shanten()
    print('all eval_v2 tests passed')
