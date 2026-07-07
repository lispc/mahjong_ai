# cython: language_level=3
"""Cython 加速的 eval0/eval1/eval2 核心。

新增 eval2_metric_counts：输入 34 维 counts 和 34 维 tile 概率，
直接返回 eval2 metric，全部在 C/C++ 中完成，避免 Python 递归开销。
"""
from libcpp.vector cimport vector


cdef void _search_suit(int* counts, int start, int i, int melds, int pairs,
                       vector[int]* m_out, vector[int]* p_out) noexcept nogil:
    """递归搜索一个花色的面子/搭子分解，记录所有可达 (melds, pairs)。"""
    while i < 9 and counts[start + i] == 0:
        i += 1
    if i >= 9:
        m_out.push_back(melds)
        p_out.push_back(pairs)
        return

    cdef int cnt = counts[start + i]

    # 刻子
    if cnt >= 3:
        counts[start + i] -= 3
        _search_suit(counts, start, i, melds + 1, pairs, m_out, p_out)
        counts[start + i] += 3

    # 顺子
    if i < 7 and counts[start + i + 1] > 0 and counts[start + i + 2] > 0:
        counts[start + i] -= 1
        counts[start + i + 1] -= 1
        counts[start + i + 2] -= 1
        _search_suit(counts, start, i, melds + 1, pairs, m_out, p_out)
        counts[start + i] += 1
        counts[start + i + 1] += 1
        counts[start + i + 2] += 1

    # 对子
    if cnt >= 2:
        counts[start + i] -= 2
        _search_suit(counts, start, i, melds, pairs + 1, m_out, p_out)
        counts[start + i] += 2

    # 跳过
    _search_suit(counts, start, i + 1, melds, pairs, m_out, p_out)


cdef void _suit_frontier(int* counts, int start,
                         vector[int]* m_out, vector[int]* p_out) noexcept nogil:
    _search_suit(counts, start, 0, 0, 0, m_out, p_out)


cdef void _honors_frontier(int* counts, int start,
                           vector[int]* m_out, vector[int]* p_out) noexcept nogil:
    cdef int melds = 0
    cdef int pairs = 0
    cdef int i, c
    for i in range(7):
        c = counts[start + i]
        melds += c // 3
        c = c % 3
        if c >= 2:
            pairs += 1
    m_out.push_back(melds)
    p_out.push_back(pairs)


cdef void _pareto_strip(vector[int]* m_list, vector[int]* p_list) noexcept nogil:
    """对 (melds, pairs) 列表做 Pareto strip：保留不被支配的点。"""
    cdef int n = m_list.size()
    if n == 0:
        return
    cdef vector[int] new_m
    cdef vector[int] new_p
    cdef int i, j
    cdef bint skip
    for i in range(n):
        skip = False
        for j in range(n):
            if i == j:
                continue
            if m_list[0][j] >= m_list[0][i] and p_list[0][j] >= p_list[0][i]:
                if m_list[0][j] > m_list[0][i] or p_list[0][j] > p_list[0][i]:
                    skip = True
                    break
        if not skip:
            new_m.push_back(m_list[0][i])
            new_p.push_back(p_list[0][i])
    m_list[0] = new_m
    p_list[0] = new_p


cdef double _merge_two_frontiers(vector[int]* m1, vector[int]* p1,
                                 vector[int]* m2, vector[int]* p2,
                                 vector[int]* out_m, vector[int]* out_p) noexcept nogil:
    """合并两组 frontier，输出 Pareto frontier。"""
    cdef int i, j
    out_m.clear()
    out_p.clear()
    for i in range(m1.size()):
        for j in range(m2.size()):
            out_m.push_back(m1[0][i] + m2[0][j])
            out_p.push_back(p1[0][i] + p2[0][j])
    _pareto_strip(out_m, out_p)
    return 0.0


cdef double _metric_from_counts(int* counts, double pair_coef) noexcept nogil:
    """从 34 维 counts 计算最大 metric。"""
    cdef vector[int] m0, p0, m1, p1, m2, p2, m3, p3
    cdef vector[int] tmp_m, tmp_p, cur_m, cur_p
    cdef int i

    _suit_frontier(counts, 0, &m0, &p0)
    _suit_frontier(counts, 9, &m1, &p1)
    _suit_frontier(counts, 18, &m2, &p2)
    _honors_frontier(counts, 27, &m3, &p3)

    _merge_two_frontiers(&m0, &p0, &m1, &p1, &cur_m, &cur_p)
    _merge_two_frontiers(&cur_m, &cur_p, &m2, &p2, &tmp_m, &tmp_p)
    cur_m = tmp_m
    cur_p = tmp_p
    _merge_two_frontiers(&cur_m, &cur_p, &m3, &p3, &tmp_m, &tmp_p)
    cur_m = tmp_m
    cur_p = tmp_p

    cdef double best = 0.0
    cdef double metric
    for i in range(cur_m.size()):
        metric = cur_m[i] + (1.0 if cur_p[i] > 0 else 0.0) * pair_coef
        if metric > best:
            best = metric
    return best


def eval0_metric_counts(const int[::1] counts not None, double pair_coef=1.0) -> float:
    """输入 34 维 counts，返回 metric。"""
    cdef int buf[34]
    cdef int i
    for i in range(34):
        buf[i] = counts[i]
    return float(_metric_from_counts(buf, pair_coef))


def eval0_metric_tiles(list tiles not None, double pair_coef=1.0) -> float:
    """输入 tile 列表，返回 metric。"""
    cdef int counts[34]
    cdef int i
    for i in range(34):
        counts[i] = 0
    cdef int t
    for t in tiles:
        if 1 <= t <= 9:
            counts[t - 1] += 1
        elif 11 <= t <= 19:
            counts[t - 11 + 9] += 1
        elif 21 <= t <= 29:
            counts[t - 21 + 18] += 1
        elif 31 <= t <= 37:
            counts[t - 31 + 27] += 1
    return float(_metric_from_counts(counts, pair_coef))


cdef double _eval1_from_counts(int* counts, int* used, double pair_coef) noexcept nogil:
    """eval1 = sum_k p[k] * eval0(counts + k)，概率基于 remaining = all - used - counts。"""
    cdef double total = 0.0
    cdef double rem_sum = 0.0
    cdef double rem[34]
    cdef int k
    for k in range(34):
        rem[k] = 4.0 - used[k] - counts[k]  # 每种牌默认 4 张
        if rem[k] < 0.0:
            rem[k] = 0.0
        rem_sum += rem[k]
    if rem_sum <= 0.0:
        return _metric_from_counts(counts, pair_coef)

    cdef double p
    for k in range(34):
        p = rem[k] / rem_sum
        if p <= 0.0:
            continue
        counts[k] += 1
        total += p * _metric_from_counts(counts, pair_coef)
        counts[k] -= 1
    return total


cdef double _eval2_from_counts(int* counts, int* used, double pair_coef) noexcept nogil:
    """eval2 = sum_k p[k] * eval1(counts + k)。"""
    cdef double total = 0.0
    cdef double rem_sum = 0.0
    cdef double rem[34]
    cdef int k
    for k in range(34):
        rem[k] = 4.0 - used[k] - counts[k]
        if rem[k] < 0.0:
            rem[k] = 0.0
        rem_sum += rem[k]
    if rem_sum <= 0.0:
        return _metric_from_counts(counts, pair_coef)

    cdef double p
    for k in range(34):
        p = rem[k] / rem_sum
        if p <= 0.0:
            continue
        counts[k] += 1
        total += p * _eval1_from_counts(counts, used, pair_coef)
        counts[k] -= 1
    return total


def eval2_metric_counts(const int[::1] counts not None,
                        const int[::1] used not None,
                        double pair_coef=1.0) -> float:
    """输入 34 维 counts 和 34 维 used（已见张数），返回 eval2 metric。"""
    cdef int buf[34]
    cdef int ubuf[34]
    cdef int i
    for i in range(34):
        buf[i] = counts[i]
        ubuf[i] = used[i]
    return float(_eval2_from_counts(buf, ubuf, pair_coef))


def eval2_metric_tiles(list tiles not None,
                       list used_tiles not None,
                       double pair_coef=1.0) -> float:
    """输入 tile 列表（手牌）和 used_tiles 列表（已见牌），返回 eval2 metric。"""
    cdef int counts[34]
    cdef int used[34]
    cdef int i
    for i in range(34):
        counts[i] = 0
        used[i] = 0
    cdef int t
    for t in tiles:
        if 1 <= t <= 9:
            counts[t - 1] += 1
        elif 11 <= t <= 19:
            counts[t - 11 + 9] += 1
        elif 21 <= t <= 29:
            counts[t - 21 + 18] += 1
        elif 31 <= t <= 37:
            counts[t - 31 + 27] += 1
    for t in used_tiles:
        if 1 <= t <= 9:
            used[t - 1] += 1
        elif 11 <= t <= 19:
            used[t - 11 + 9] += 1
        elif 21 <= t <= 29:
            used[t - 21 + 18] += 1
        elif 31 <= t <= 37:
            used[t - 31 + 27] += 1
    return float(_eval2_from_counts(counts, used, pair_coef))
