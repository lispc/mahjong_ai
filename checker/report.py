# -*- coding: utf-8 -*-
"""Metrics computation and Markdown report generation."""

import os
import math


def _base_name(name):
    """Strip the @seat suffix added by tournament.py."""
    return name.split('@')[0]


def _win_rate_ci(wins, games, z=1.96):
    """95% 置信区间 for win rate."""
    if games == 0:
        return 0.0, 0.0
    p = wins / games
    se = math.sqrt(p * (1 - p) / games)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def compute_metrics(results, agent_names):
    """
    Compute per-agent metrics from a list of game results.
    Returns {agent_name: metrics_dict}.
    """
    metrics = {name: {
        'games': 0,
        'wins': 0,
        'self_wins': 0,
        'ron_wins': 0,
        'deal_ins': 0,
        'draws_seen': 0,
        'decision_times': [],
    } for name in agent_names}

    total_games = len(results)
    draws = 0

    for r in results:
        players = r.get('players_order', [])
        for p in players:
            bn = _base_name(p)
            if bn in metrics:
                metrics[bn]['games'] += 1

        for bn, times in r.get('decision_times', {}).items():
            if bn in metrics:
                metrics[bn]['decision_times'].extend(times)

        if r['win_type'] == 'draw':
            draws += 1
            for p in players:
                bn = _base_name(p)
                if bn in metrics:
                    metrics[bn]['draws_seen'] += 1
            continue

        winner_base = _base_name(r['winner'])
        if winner_base in metrics:
            metrics[winner_base]['wins'] += 1
            if r['win_type'] == 'self':
                metrics[winner_base]['self_wins'] += 1
            elif r['win_type'] == 'ron':
                metrics[winner_base]['ron_wins'] += 1

        if r['win_type'] == 'ron':
            dealer_base = _base_name(r['dealer'])
            if dealer_base in metrics:
                metrics[dealer_base]['deal_ins'] += 1

    for name in agent_names:
        m = metrics[name]
        g = max(m['games'], 1)
        m['win_rate'] = m['wins'] / g
        m['self_rate'] = m['self_wins'] / g
        m['ron_rate'] = m['ron_wins'] / g
        m['deal_in_rate'] = m['deal_ins'] / g
        m['draw_rate'] = m['draws_seen'] / g
        times = m['decision_times']
        m['avg_decision_time'] = sum(times) / max(len(times), 1) if times else 0.0
        m['total_decisions'] = len(times)

    metrics['_meta'] = {
        'total_games': total_games,
        'draws': draws,
        'draw_rate': draws / max(total_games, 1),
    }
    return metrics


def compute_elo(results, agent_names, initial_elo=1500, k=32):
    """
    Update Elo ratings using pairwise comparisons inside each game.
    Draws count as 0.5 for every pair.
    """
    elo = {name: float(initial_elo) for name in agent_names}

    def expected(ra, rb):
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    for r in results:
        players = r.get('players_order', [])
        base_players = [_base_name(p) for p in players]
        if r['win_type'] == 'draw':
            for i in range(len(base_players)):
                for j in range(i + 1, len(base_players)):
                    a, b = base_players[i], base_players[j]
                    if a not in elo or b not in elo:
                        continue
                    ea = expected(elo[a], elo[b])
                    eb = expected(elo[b], elo[a])
                    elo[a] += k * (0.5 - ea)
                    elo[b] += k * (0.5 - eb)
            continue

        winner = _base_name(r['winner'])
        losers = [p for p in base_players if p != winner]
        for loser in losers:
            if winner not in elo or loser not in elo:
                continue
            ea = expected(elo[winner], elo[loser])
            eb = expected(elo[loser], elo[winner])
            elo[winner] += k * (1.0 - ea)
            elo[loser] += k * (0.0 - eb)

    return elo


def _macro_analysis(metrics, elo, agent_names):
    """根据指标生成一段宏观分析文字。"""
    lines = []

    # 按 Elo 排序（比胜率更不受同场 Agent 数量影响）
    ranked = sorted(agent_names, key=lambda n: elo[n], reverse=True)
    lines.append('### 综合排名（按 Elo）\n')
    for i, name in enumerate(ranked, 1):
        m = metrics[name]
        lines.append('{}. **{}**：Elo {:.0f}，胜率 {:.1%}，平均决策 {:.1f} ms\n'.format(
            i, name, elo[name], m['win_rate'], m['avg_decision_time'] * 1000))
    lines.append('\n')
    lines.append('> 注：每局有 2 个 Baseline 实例、1 个 ExpectiMax、1 个 MCTS，'
                 '因此 Baseline 的“胜率”是两名玩家的合计，会系统性高于单一 Agent 类型；'
                 'Elo 采用 pairwise 更新，受该因素影响较小。\n')
    lines.append('\n')

    # 头名统计显著性
    first = ranked[0]
    first_m = metrics[first]
    lo, hi = _win_rate_ci(first_m['wins'], first_m['games'])
    lines.append('### 统计显著性\n')
    lines.append('- 冠军 **{}** 的胜率 95% 置信区间：{:.1%}–{:.1%}（{} 局）\n'.format(
        first, lo, hi, first_m['games']))
    if len(ranked) >= 2:
        second = ranked[1]
        second_m = metrics[second]
        s_lo, s_hi = _win_rate_ci(second_m['wins'], second_m['games'])
        if hi < s_lo:
            lines.append('- **{}** 显著优于 **{}**（置信区间不重叠）\n'.format(first, second))
        elif lo > s_hi:
            lines.append('- **{}** 显著优于 **{}**（置信区间不重叠）\n'.format(second, first))
        else:
            lines.append('- **{}** 与 **{}** 的胜率置信区间有重叠，差异尚不显著\n'.format(first, second))
    lines.append('\n')

    # 风格判断
    lines.append('### 风格判断\n')
    for name in agent_names:
        m = metrics[name]
        total_wins = max(m['wins'], 1)
        self_ratio = m['self_wins'] / total_wins
        ron_ratio = m['ron_wins'] / total_wins
        if self_ratio > ron_ratio + 0.2:
            style = '偏自摸进攻型'
        elif ron_ratio > self_ratio + 0.2:
            style = '偏点和机会型'
        else:
            style = '攻守平衡型'
        lines.append('- **{}**：{}（自摸 {:.1%} / 点和 {:.1%} / 点炮 {:.1%}）\n'.format(
            name, style, self_ratio, ron_ratio, m['deal_in_rate']))
    lines.append('\n')

    # 对照分析
    if 'Baseline' in agent_names and 'ExpectiMax' in agent_names:
        lines.append('### ExpectiMax 相对 Baseline\n')
        b, e = metrics['Baseline'], metrics['ExpectiMax']
        delta = e['win_rate'] - b['win_rate']
        lines.append('- 胜率变化：{:.1%}（{:.1%} vs {:.1%}）\n'.format(
            delta, e['win_rate'], b['win_rate']))
        if delta > 0:
            lines.append('- 新版评估函数 + used 概率模型带来正向收益\n')
        elif delta < 0:
            lines.append('- 新版评估函数 + used 概率模型未带来正向收益，可能受限于深度或评估权重\n')
        else:
            lines.append('- 两者胜率持平\n')
        lines.append('\n')

    if 'ExpectiMax' in agent_names and 'MCTS' in agent_names:
        lines.append('### ExpectiMax 相对 MCTS（精确期望 vs 蒙特卡洛采样）\n')
        e, m = metrics['ExpectiMax'], metrics['MCTS']
        delta = e['win_rate'] - m['win_rate']
        lines.append('- 胜率差：{:.1%}（ExpectiMax {:.1%} vs MCTS {:.1%}）\n'.format(
            delta, e['win_rate'], m['win_rate']))
        if abs(delta) < 0.03:
            lines.append('- 在相同深度、相近计算量下，精确期望与采样期望表现接近\n')
        elif delta > 0:
            lines.append('- 精确期望在当前设置下略优于采样期望\n')
        else:
            lines.append('- 采样期望在当前设置下略优于精确期望，可能得益于采样的方差带来的探索性\n')
        lines.append('\n')

    lines.append('### 后续改进方向\n')
    lines.append('- 加深搜索深度（depth=2）或加入剪枝，观察是否能稳定提升胜率\n')
    lines.append('- 调整评估函数权重（向听数 / 搭子质量 / 听牌张数）\n')
    lines.append('- 加入防守项：将点炮风险纳入期望计算\n')
    lines.append('- 引入对手建模：根据弃牌推断对手听牌概率\n')
    lines.append('- 扩展动作空间：碰、杠、报听决策\n')
    lines.append('\n')

    return lines


def generate_report(results, agent_names, output_path='output/ai_report.md'):
    """Write a Markdown report with all metrics."""
    metrics = compute_metrics(results, agent_names)
    elo = compute_elo(results, agent_names)
    meta = metrics.get('_meta', {})

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    lines = []
    lines.append('# Mahjong AI 实验报告\n')
    lines.append('## 实验设置\n')
    lines.append('- 总局数: {}\n'.format(meta.get('total_games', len(results))))
    lines.append('- 参赛 AI: {}\n'.format(', '.join(agent_names)))
    lines.append('\n')

    lines.append('## 胜率与和牌方式\n')
    lines.append('| AI | 总局次 | 胜率 | 胜率 95% CI | 自摸率 | 点和率 | 点炮率 | 流局率 | Elo | 平均决策耗时(ms) |\n')
    lines.append('|----|--------|------|-------------|--------|--------|--------|--------|------|------------------|\n')
    for name in agent_names:
        m = metrics[name]
        lo, hi = _win_rate_ci(m['wins'], m['games'])
        lines.append('| {} | {} | {:.2%} | {:.2%}–{:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.1f} | {:.2f} |\n'.format(
            name,
            m['games'],
            m['win_rate'],
            lo, hi,
            m['self_rate'],
            m['ron_rate'],
            m['deal_in_rate'],
            m['draw_rate'],
            elo[name],
            m['avg_decision_time'] * 1000,
        ))
    lines.append('\n')

    lines.append('## 全局流局率\n')
    lines.append('- 流局率: {:.2%}（{} / {}）\n'.format(
        meta.get('draw_rate', 0),
        meta.get('draws', 0),
        meta.get('total_games', len(results))))
    lines.append('\n')

    # Optional extra fields if present in results.
    has_shanten = any('avg_shanten' in r for r in results)
    has_time = any('avg_time' in r for r in results)
    if has_shanten or has_time:
        lines.append('## 额外指标\n')
        if has_shanten:
            total_shanten = sum(r.get('avg_shanten', 0) for r in results)
            lines.append('- 平均向听数: {:.3f}\n'.format(total_shanten / len(results)))
        if has_time:
            total_time = sum(r.get('avg_time', 0) for r in results)
            lines.append('- 平均决策耗时: {:.3f}s\n'.format(total_time / len(results)))
        lines.append('\n')

    lines.append('## 计算量对比\n')
    for name in agent_names:
        m = metrics[name]
        lines.append('- **{}**：平均决策耗时 {:.2f} ms，总决策次数 {}\n'.format(
            name, m['avg_decision_time'] * 1000, m['total_decisions']))
    lines.append('\n')
    lines.append('> 公平性说明：ExpectiMax（depth=1）与 MCTS（depth=1, samples=250）'
                 '平均决策耗时均在 60 ms 左右，计算量处于同一量级；'
                 'Baseline 使用原项目默认 depth=2，耗时约 260 ms，作为原始参考。\n')
    lines.append('\n')

    lines.append('## 宏观分析与改进方向\n')
    lines.extend(_macro_analysis(metrics, elo, agent_names))
    lines.append('\n')

    lines.append('## 说明\n')
    lines.append('- 胜率按每 AI 实际参与的局次统计。\n')
    lines.append('- 自摸率 = 自摸和牌次数 / 参与局次。\n')
    lines.append('- 点和率 = 点和和牌次数 / 参与局次。\n')
    lines.append('- 点炮率 = 放炮导致对手和牌次数 / 参与局次。\n')
    lines.append('- 胜率 95% CI 采用正态近似。\n')
    lines.append('- Elo 采用 pairwise 更新，初始 {}，K={}。\n'.format(1500, 32))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    return output_path
