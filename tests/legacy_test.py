import sys
sys.path.insert(0, '/Users/zhangzhuo/repos/personal/mahjong_ai')

from tile import *
from utils import *
from algo import *
import time


def test_eval_honors():
    assert_eq(eval_honors([east, east, east, west, west, west]), EvalResult.from_list([(2,0),(1,1),(0,2)]))
    cnt = 0
    while cnt != 0:
        cnt -= 1
        tiles = rand_seq()
        print(display_tiles(tiles))
        print(eval0(tiles))


def test_scatter():
    assert_eq(scatter({'a':2,'b':2},3), [{'a':1,'b':2}, {'a':2,'b':1}])


def test_append():
    assert_eq(multi_append({(1,): 4}, 2, 4), [{(1, 2): 4}])


def test_list_sub():
    assert_eq(list_sub([1,2,2,3,3,3,4,4,4,4], [2,2,2,3,3]), [1,3,4,4,4,4])


def test_select():
    tiles = [6, 7, 8, 11, 11, 11, 12, 13, 18, 18, 22, 28, 2, 29]
    # Cython eval2 后同分候选 tie-break 顺序不稳定，按集合比较（见 AGENTS.md §7.10）
    assert_eq(set(select(tiles, with_prob=False)[:2]), {wan2, tong2})
    tiles = [2, 5, 5, 6, 11, 11, 14, 24, 26, 28, 28, 29, 33, 36]
    assert_eq(select(tiles, with_prob=False)[:2], [blank, south])
    tiles = [1, 1, 1, 7, 8, 13, 16, 16, 18, 6, 7, 8, 31, 33]
    assert_eq(select(tiles, with_prob=False)[:3], [south, east, suo3])
    tiles = [11, 16, 18, 18, 21, 21, 21, 21, 26, 27, 34, 36, 36, 37]
    assert_eq(select(tiles, with_prob=False)[:4], [facai, north, tong1, suo1])


def demo_select():
    def check(seq):
        print('14张牌:', seq)
        print(display_tiles(seq))
        res = select(seq)
        print('建议出牌:')
        for m, c in res[:5]:
            print(tile_to_str(c), '%.3f'%m)
        print('')
    cnt = 5
    start = time.time()
    i = 0
    while i < cnt:
        i += 1
        tiles = rand_seq_no_single(14)
        check(tiles)
    end = time.time()
    print('avg time is', (end-start)/cnt)


def test_eval_suit():
    def max3seq(l):
        return eval_suit(l).max3()
    assert_eq(max3seq([1, 2, 3, 4, 4, 5, 5, 6, 6]), 3)
    assert_eq(max3seq([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]), 4)
    assert_eq(max3seq([1, 2, 3, 5, 6, 7]), 2)
    assert_eq(max3seq([1, 1, 2, 2, 3, 3]), 2)  # 123 123
    assert_eq(max3seq([1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 9]), 3) # 123 123 345|456
    assert_eq(max3seq([wan2, wan2, wan2, wan2, wan3, wan3, wan3, wan3, wan4, wan4, wan4, wan4]), 4)
    cnt = 0
    while cnt != 0:
        cnt -= 1
        tiles = rand_single_color()
        print(sorted(tiles), max3seq(tiles))


def test_slice():
    assert_eq(to_slice(1), [
        [1]
    ])
    assert_eq(to_slice(2), [
        [1,1],
        [2]
    ])
    assert_eq(to_slice(3), [
        [1,1,1],
        [2,1],
        [3]
    ])
    assert_eq(to_slice(4), [
                [1,1,1,1],
                [2,1,1],
                [2,2],
                [3,1]
    ])


def test_is_succ():
    data = [wan4, wan5, wan6, suo1, suo1, suo5, suo6, suo7, suo8, tong1, tong3, tong5, tong6, tong7]
    assert_eq(is_succ(data), False)
    data = [6, 7, 8, 11, 11, 13, 15, 16, 17, 21, 21, 26, 27, 28]
    assert_eq(is_succ(data), False)


if __name__ == '__main__':
    test_is_succ()
    test_list_sub()
    test_slice()
    test_scatter()
    test_append()
    test_eval_suit()
    test_eval_honors()
    test_select()
    demo_select()
