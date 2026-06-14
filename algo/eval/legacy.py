from collections import defaultdict
import copy
from utils import *
import tile
import operator
import config
import context


# 合法牌 id 有序列表（34 种）
_TILE_IDS = list(range(1, 10)) + list(range(11, 20)) + list(range(21, 30)) + list(range(31, 38))

# eval0 的默认 Context 单例 sentinel
_DEFAULT_CONTEXT = context.Context()


class TilePool:
    def __init__(self):
        self.pool = count(tile.all_tiles())
        self.num = sum(self.pool.values())

    def consume(self, t):
        if t not in self.pool:
            raise "invalid tile " + str(t)
        if self.pool[t] == 1:
            del self.pool[t]
        else:
            self.pool[t] -= 1
        self.num -= 1

    def consume_multi(self, l):
        for item in l:
            self.consume(item)


def scatter(slots, num):
    if sum(slots.values()) < num:
        return []
    if num == 0:
        return [{k:0 for k in slots}]
    if len(slots) == 0:
        return [{}]
    keys = list(slots.keys())
    results = []
    first_key = keys[0]
    for i in range(0,slots[first_key]+1):  # how many to put in this slot
        if i > num:
            break
        new_slots = {k:slots[k] for k in slots if k != first_key}
        new_num = num - i
        res = scatter(new_slots, new_num)
        for item in res:
            item[first_key] = i
            results.append(item)
    return results


def multi_append(base, item_to_add, item_num):
    if item_num == 0:
        return [copy.deepcopy(base)]
    base_num = sum(base.values())
    if base_num <= item_num:
        result = {k+(item_to_add,): v for k, v in base.items()}
        if base_num < item_num:
            result[(item_to_add,)] = item_num - base_num
        return [result]
    else:
        #keys_num = len(base)
        results = []
        res = scatter(base, item_num)
        for item in res:
            base_clone = copy.deepcopy(base)
            for k, v in item.items():
                if v != 0:
                    base_clone[k + (item_to_add,)] = v
                    base_clone[k] -= v
                    if base_clone[k] == 0:
                        del base_clone[k]
            results.append(base_clone)
        return results


class State:
    def __init__(self):
        self.active = {} # tuple -> count
        self.non_active = {}
        self.parent = None

    def __repr__(self):
        return '(' + repr(self.active) + ',' + repr(self.non_active) + ')'

    def strip_dead(self, active_idx):
        for item in list(self.active.keys()):
            if item[-1] != active_idx:
                del self.active[item]

    def freeze(self):
        #print('freeze', repr(self), 'parent', repr(self.parent))
        for item in list(self.active.keys()):
            if len(item) == 3:
                assert item not in self.non_active, repr(item) + ' should not in ' + repr(self.non_active)
                assert self.active[item] != 0
                self.non_active[item] = self.active[item]
                del self.active[item]
        #print('freeze to', repr(self))


    def transit(self, tile, num):
        self.strip_dead(tile-1)
        slices = to_slice(num)
        results = []
        for ptn in slices:
            single_items = [item for item in ptn if item == 1]
            multi_items = [item for item in ptn if item != 1]
            to_add_num = len(single_items)
            new_non_active = copy.deepcopy(self.non_active)
            for n in multi_items:
                k = (tile,) * n
                if k not in new_non_active:
                    new_non_active[k] = 0
                new_non_active[k] += 1
            for x in multi_append(self.active, tile, to_add_num):
                new_item = State()
                new_item.active = x
                new_item.non_active = copy.deepcopy(new_non_active)
                new_item.parent = self
                new_item.freeze()
                new_item.strip_dead(tile)
                results.append(new_item)
        return results


def eval_suit(l):
    r = EvalResult()
    last_tile = None
    seq_count = ()
    for t, num in sorted(count(l).items()):
        if last_tile is not None and t - last_tile == 1:
                last_tile = t
                seq_count = seq_count + (num,)
                continue
        else:
            r.merge(eval_seq(seq_count))
            last_tile = t
            seq_count = (num,)
    r.merge(eval_seq(seq_count))
    return r


@cache_f
def eval_seq(c):
    states = [State()]
    for idx, num in enumerate(c):
        new_states = []
        for s in states:
            new_states += s.transit(idx, num)
        states = new_states
    result = []
    for item in states:
        two_count = 0
        three_count = 0
        for k, v in item.non_active.items():
            if len(k) == 2:
                two_count += v
            elif len(k) == 3:
                three_count += v
            else:
                assert False, 'wtf' + repr(item)
        result.append((three_count, two_count))
    return EvalResult.from_list(result)


def to_slice(i, max_len=3):
    if i == 0:
        return [[]]
    result = []
    for l in range(1, min(max_len, i)+1):
        result += [[l] + item for item in to_slice(i-l,l)]
    return result


def eval_one_honor_tile(n):
    # 健壮性： Honor 数量不会超过 4，若出现异常输入按 4 处理
    if n >= 4:
        return EvalResult.from_list([(1,0),(0,2)])
    if n == 3:
        return EvalResult.from_list([(1,0),(0,1)])
    if n == 2:
        return EvalResult.from_list([(0,1)])
    return EvalResult.from_list([(0,0)])


class EvalLeafResult:
    def __init__(self, melds_num=0, pairs_num=0):
        self.melds_num = melds_num
        self.pairs_num = pairs_num

    def __add__(self, other):
        return EvalLeafResult(self.melds_num + other.melds_num, self.pairs_num + other.pairs_num)

    def __lt__(self, other):
        if self.melds_num < other.melds_num:
            return True
        if self.melds_num == other.melds_num and self.pairs_num < other.pairs_num:
            return True
        return False

    def __eq__(self, other):
        return self.melds_num == other.melds_num and self.pairs_num == other.pairs_num

    def metric(self):
        return self.melds_num + config.pair_coef * (min(self.pairs_num, 1))

    def is_succ(self):
        return self.melds_num == 4 and self.pairs_num == 1

    def __repr__(self):
        return '(%d,%d)'%(self.melds_num, self.pairs_num)


class EvalResult:
    def __init__(self):
        self.leafs = [EvalLeafResult()]

    def merge(self, other):
        self.leafs = EvalResult.strip(join_list(self.leafs, other.leafs))

    def metric(self):
        return sorted([item.metric() for item in self.leafs], reverse=True)[0]

    def max3(self):
        return self.leafs[0].melds_num

    @staticmethod
    def strip(x):
        s = sorted(x, reverse=True)
        result = [s[0]]
        for item in s[1:]:
            skip = False
            for c in result:
                if c.melds_num >= item.melds_num and c.pairs_num >= item.pairs_num:
                    skip = True
                    break
            if not skip:
                result.append(item)
        return result

    @staticmethod
    def from_list(x):
        r = EvalResult()
        r.leafs = [EvalLeafResult(m, p) for m, p in sorted(list(set(x)), reverse=True)]
        return r

    def __repr__(self):
        return '['+','.join(map(repr, self.leafs))+']'

    def __eq__(self, other):
        return self.leafs == other.leafs

    def is_succ(self):
        return any(item.is_succ() for item in self.leafs)


def eval_honors(l):
    c = count(l)
    result = EvalResult()
    for k, v in c.items():
        result.merge(eval_one_honor_tile(v))
    return result


def eval_naive(l):
    splits = tile.split_by_category(l)
    result = EvalResult()
    for item in splits[:3]:
        result.merge(eval_suit(item))
    result.merge(eval_honors(splits[-1]))
    return result


def is_succ(l):
    return eval_naive(l).is_succ()


_eval0_cache = {}


def _eval0_key(tiles):
    """用于 eval0 缓存的规范 key：34 维数量元组。"""
    c = count(tiles)
    return tuple(c.get(t, 0) for t in _TILE_IDS)


def eval0(l, c=_DEFAULT_CONTEXT):
    key = _eval0_key(l)
    result = _eval0_cache.get(key)
    if result is not None:
        return result
    result = eval_naive(l).metric()
    _eval0_cache[key] = result
    return result


def eval_rec(tiles, f, c=_DEFAULT_CONTEXT, verbose=False):
    if verbose:
        print('所有牌')
        print(tile.display_tiles(tiles))

    # 避免原地修改默认的模块级 Context 单例
    if c is _DEFAULT_CONTEXT:
        c = context.Context()

    prob = c.tile_prob(tiles)
    base_metric = f(tiles, c)
    final_metric = 0
    for k in prob:
        # 原代码 deepcopy context 但实际 eval_rec 并不会修改 used；
        # 直接复用同一个 context 即可，避免拷贝开销。
        metric = f(tiles + [k], c)
        final_metric += prob[k] * metric
        if verbose:
            print('摸牌:', tile.tile_to_str(k), '概率', prob[k], '得分', '%.2f'%base_metric,'+', '%.2f'%(metric-base_metric))
    if verbose:
        print('最终指标', final_metric)
    return final_metric


def eval1(tiles, c=context.Context()):
    return eval_rec(tiles, eval0, c)


def eval2(tiles, c=context.Context()):
    return eval_rec(tiles, eval1, c)


def select(tiles, with_prob=True, metric_f=eval2, c=context.Context()):
    assert len(tiles) == 14
    best = []
    handled = set()
    for idx in range(len(tiles)):
        if tiles[idx] in handled:
            continue
        handled.add(tiles[idx])
        tiles_clone = tiles[:]
        del tiles_clone[idx]
        metric = metric_f(tiles_clone, c)
        best.append((metric, tiles[idx]))
    result = sorted(best, reverse=True)
    if not with_prob:
        return [item for _, item in result]
    return result

