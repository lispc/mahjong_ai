# -*- coding: utf-8 -*-
"""Self-contained Mahjong game driver.

支持完整动作空间：
- 摸牌后自摸/打牌
- 打牌后其他玩家可依次选择胡/杠/碰
- 杠后从牌山尾部补牌
- 报听后锁手
"""

from agent import Message
import tile_pool
from algo.eval.v2 import shanten

import time


def _notify_meld(agents, claimer, meld_type, tile_val):
    """通知所有玩家有副露发生。"""
    msg = Message(claimer, 'meld', {'type': meld_type, 'tile': tile_val})
    for ag in agents:
        ag.handle_msg(msg)


def _claim_peng(agent, tile_val):
    """执行碰：从闭手移除两张 tile_val，添加副露。"""
    for _ in range(2):
        agent.cur.remove(tile_val)
    agent.add_meld('peng', tile_val)


def _claim_gang(agent, tile_val):
    """执行杠：从闭手移除三张 tile_val，添加副露。"""
    for _ in range(3):
        agent.cur.remove(tile_val)
    agent.add_meld('gang', tile_val)


def _process_claims(agents, discarded, discarder_turn, pool, locked_names,
                    event_log, record_log):
    """
    处理一次弃牌后的吃/碰/杠/胡声明。
    返回 {'type': 'win'/'gang'/'peng'/'pass', 'winner': ..., 'claimer': ...}
    """
    n = len(agents)

    # 1) 胡牌：优先级最高，按逆时针顺序询问
    for offset in range(1, n):
        idx = (discarder_turn + offset) % n
        other = agents[idx]
        if other.respond_hu(discarded, getattr(other, 'context', None)):
            # 把这张牌算进和牌者的手牌（用于统计）
            other.cur.append(discarded)
            if record_log:
                event_log.append({
                    'type': 'claim',
                    'player': other.name,
                    'claim_type': 'hu',
                    'tile': discarded,
                    'wall_remaining': len(pool.tiles) - pool.idx,
                })
            return {'type': 'win', 'winner': other.name,
                    'dealer': agents[discarder_turn].name}

    # 2) 杠：其次
    for offset in range(1, n):
        idx = (discarder_turn + offset) % n
        other = agents[idx]
        if other.name in locked_names:
            continue
        if other._can_gang(discarded) and \
                other.respond_gang(discarded, getattr(other, 'context', None)):
            _claim_gang(other, discarded)
            _notify_meld(agents, other.name, 'gang', discarded)
            if record_log:
                event_log.append({
                    'type': 'claim',
                    'player': other.name,
                    'claim_type': 'gang',
                    'tile': discarded,
                    'wall_remaining': len(pool.tiles) - pool.idx,
                })
            return {'type': 'gang', 'claimer': idx}

    # 3) 碰：最后
    for offset in range(1, n):
        idx = (discarder_turn + offset) % n
        other = agents[idx]
        if other.name in locked_names:
            continue
        if other._can_peng(discarded) and \
                other.respond_peng(discarded, getattr(other, 'context', None)):
            _claim_peng(other, discarded)
            _notify_meld(agents, other.name, 'peng', discarded)
            if record_log:
                event_log.append({
                    'type': 'claim',
                    'player': other.name,
                    'claim_type': 'peng',
                    'tile': discarded,
                    'wall_remaining': len(pool.tiles) - pool.idx,
                })
            return {'type': 'peng', 'claimer': idx}

    return {'type': 'pass'}


def _discard_step(current, drawn, locked_names, record_time, decision_times,
                  record_log, event_log, wall_remaining_fn):
    """
    处理当前玩家摸牌后的出牌阶段。
    返回 (discarded, dt, locked)。
    """
    if current.name in locked_names:
        # 报听后锁死：摸到非胡牌必须原样打出
        discarded = drawn
        current.cur.remove(discarded)
        dt = 0.0
        locked = True
    else:
        t0 = time.time()
        discarded = current.next()
        dt = time.time() - t0
        if record_time:
            bn = current.name.split('@')[0]
            decision_times.setdefault(bn, []).append(dt)
        locked = False

    if record_log:
        event_log.append({
            'type': 'discard',
            'player': current.name,
            'tile': discarded,
            'locked': locked,
            'decision_time': round(dt, 4),
            'wall_remaining': wall_remaining_fn(),
        })
    return discarded, dt, locked


def _tenpai_check(current, locked_names, locked, event_log, record_log,
                  wall_remaining_fn):
    """报听检测与锁手。"""
    if (not locked and current.name not in locked_names and
            len(current.full_hand()) == 13):
        if shanten(current.full_hand()) == 0:
            if current.declare_tenpai(current.cur, getattr(current, 'context', None)):
                locked_names.add(current.name)
                if record_log:
                    event_log.append({
                        'type': 'tenpai',
                        'player': current.name,
                        'wall_remaining': wall_remaining_fn(),
                    })
                tenpai_msg = Message(current.name, 'tenpai', None)
                # 通知其他玩家
                for other in (a for a in [current] if hasattr(a, 'handle_msg')):
                    pass
                for ag in []:
                    pass
                # 基类 handle_msg 只处理 put/meld；这里单独维护 tenpai 消息
                for ag in []:
                    pass
                # 实际上 BeliefExpectimaxV3Agent.handle_msg 会处理 tenpai
                # 为保持简洁，直接发给所有 agent
                for ag in [current]:
                    if hasattr(ag, 'context') and hasattr(ag.context, 'declare_tenpai'):
                        ag.context.declare_tenpai(current.name)


def play_game(agents, tile_pool_cls=None, verbose=False, record_time=False,
              record_log=False, state_callback=None):
    """
    Play one game with four agents.

    Returns a dict:
        {
            'winner': player_name or None (draw),
            'win_type': 'self' | 'ron' | 'draw',
            'players_order': [name, name, name, name],
            'decision_times': {agent_base_name: [t1, t2, ...]} (optional),
            'event_log': [...] (optional, JSON-serializable events),
        }
    """
    if tile_pool_cls is None:
        tile_pool_cls = tile_pool.Pool
    pool = tile_pool_cls()

    for agent in agents:
        agent.init_tiles(pool.next_n(13))

    def wall_remaining():
        return len(pool.tiles) - pool.idx

    event_log = []
    if record_log:
        event_log.append({
            'type': 'init',
            'players': [a.name for a in agents],
            'hands': {a.name: sorted(a.cur) for a in agents},
            'wall_remaining': wall_remaining(),
        })

    num_agents = len(agents)
    turn = 0
    decision_times = {}
    locked_names = set()
    skip_draw = False          # 碰后不需要摸牌，直接打牌
    replacement_draw = False   # 杠后需要从尾部补牌

    while True:
        current = agents[turn]

        if not skip_draw:
            if replacement_draw:
                drawn = pool.draw_replacement()
                replacement_draw = False
            else:
                drawn = pool.next()

            if drawn is None:
                result = {
                    'winner': None,
                    'win_type': 'draw',
                    'players_order': [a.name for a in agents],
                }
                if record_time:
                    result['decision_times'] = decision_times
                if record_log:
                    event_log.append({
                        'type': 'draw_end',
                        'wall_remaining': wall_remaining(),
                    })
                    result['event_log'] = event_log
                return result

            if record_log:
                event_log.append({
                    'type': 'draw',
                    'player': current.name,
                    'tile': drawn,
                    'wall_remaining': wall_remaining(),
                })

            if current.add(drawn):
                result = {
                    'winner': current.name,
                    'win_type': 'self',
                    'players_order': [a.name for a in agents],
                }
                if record_time:
                    result['decision_times'] = decision_times
                if record_log:
                    event_log.append({
                        'type': 'win',
                        'player': current.name,
                        'win_type': 'self',
                        'tile': drawn,
                        'wall_remaining': wall_remaining(),
                    })
                    result['event_log'] = event_log
                return result
        else:
            # 碰后直接进入打牌，不摸牌
            skip_draw = False

        if state_callback is not None:
            state_callback(agents, turn, 'decision', {
                'drawn': drawn,
                'skip_draw': skip_draw,
                'locked_names': set(locked_names),
                'wall_remaining': wall_remaining(),
            })

        discarded, dt, locked = _discard_step(
            current, drawn if not skip_draw else None, locked_names,
            record_time, decision_times, record_log, event_log, wall_remaining)

        # 报听检测
        if (not locked and current.name not in locked_names and
                len(current.full_hand()) == 13):
            if shanten(current.full_hand()) == 0:
                if current.declare_tenpai(current.cur, getattr(current, 'context', None)):
                    locked_names.add(current.name)
                    if record_log:
                        event_log.append({
                            'type': 'tenpai',
                            'player': current.name,
                            'wall_remaining': wall_remaining(),
                        })
                    tenpai_msg = Message(current.name, 'tenpai', None)
                    for other in agents:
                        other.handle_msg(tenpai_msg)

        # 处理其他玩家的声明
        claim = _process_claims(agents, discarded, turn, pool, locked_names,
                                event_log, record_log)

        if claim['type'] == 'win':
            result = {
                'winner': claim['winner'],
                'win_type': 'ron',
                'dealer': claim['dealer'],
                'players_order': [a.name for a in agents],
            }
            if record_time:
                result['decision_times'] = decision_times
            if record_log:
                event_log.append({
                    'type': 'win',
                    'player': claim['winner'],
                    'win_type': 'ron',
                    'tile': discarded,
                    'dealer': claim['dealer'],
                    'wall_remaining': wall_remaining(),
                })
                result['event_log'] = event_log
            return result

        if claim['type'] == 'gang':
            turn = claim['claimer']
            replacement_draw = True
            continue

        if claim['type'] == 'peng':
            turn = claim['claimer']
            skip_draw = True
            continue

        # 无人声明：正常传递 turn，并通知其他玩家这张牌进入弃牌
        turn = (turn + 1) % num_agents
        msg = Message(current.name, 'put', discarded)
        for other in agents:
            if other.name == current.name:
                continue
            other.handle_msg(msg)


def play_game_from_state(agents, wall, start_turn=0, locked_names=None,
                         verbose=False, record_time=False, record_log=False):
    """
    从指定状态继续模拟一局。

    agents: 已经初始化好手牌（init_tiles 已调用）的 agent 列表。
    wall: 剩余牌山列表，按摸牌顺序排列。
    start_turn: 下一个摸牌玩家的索引。
    locked_names: 已经报听锁手的玩家名字集合。

    返回值与 play_game 相同。
    """
    if locked_names is None:
        locked_names = set()
    wall = list(wall)
    wall_idx = 0

    def wall_remaining():
        return len(wall) - wall_idx

    event_log = []
    if record_log:
        event_log.append({
            'type': 'state',
            'players': [a.name for a in agents],
            'hands': {a.name: sorted(a.cur) for a in agents},
            'wall_remaining': wall_remaining(),
        })

    num_agents = len(agents)
    turn = start_turn % num_agents
    decision_times = {}
    skip_draw = False
    replacement_draw = False

    while True:
        current = agents[turn]

        if not skip_draw:
            if replacement_draw:
                if wall_idx >= len(wall):
                    drawn = None
                else:
                    drawn = wall.pop()
                replacement_draw = False
            else:
                if wall_idx >= len(wall):
                    drawn = None
                else:
                    drawn = wall[wall_idx]
                    wall_idx += 1

            if drawn is None:
                result = {
                    'winner': None,
                    'win_type': 'draw',
                    'players_order': [a.name for a in agents],
                }
                if record_time:
                    result['decision_times'] = decision_times
                if record_log:
                    event_log.append({
                        'type': 'draw_end',
                        'wall_remaining': wall_remaining(),
                    })
                    result['event_log'] = event_log
                return result

            if current.add(drawn):
                result = {
                    'winner': current.name,
                    'win_type': 'self',
                    'players_order': [a.name for a in agents],
                }
                if record_time:
                    result['decision_times'] = decision_times
                return result
        else:
            skip_draw = False

        discarded, dt, locked = _discard_step(
            current, drawn if not skip_draw else None, locked_names,
            record_time, decision_times, record_log, event_log, wall_remaining)

        if (not locked and current.name not in locked_names and
                len(current.full_hand()) == 13):
            if shanten(current.full_hand()) == 0:
                if current.declare_tenpai(current.cur, getattr(current, 'context', None)):
                    locked_names.add(current.name)
                    tenpai_msg = Message(current.name, 'tenpai', None)
                    for other in agents:
                        other.handle_msg(tenpai_msg)

        claim = _process_claims(agents, discarded, turn, _WallPool(wall, wall_idx),
                                locked_names, event_log, record_log)

        if claim['type'] == 'win':
            result = {
                'winner': claim['winner'],
                'win_type': 'ron',
                'dealer': claim['dealer'],
                'players_order': [a.name for a in agents],
            }
            if record_time:
                result['decision_times'] = decision_times
            if record_log:
                event_log.append({
                    'type': 'win',
                    'player': claim['winner'],
                    'win_type': 'ron',
                    'tile': discarded,
                    'dealer': claim['dealer'],
                    'wall_remaining': wall_remaining(),
                })
                result['event_log'] = event_log
            return result

        if claim['type'] == 'gang':
            turn = claim['claimer']
            replacement_draw = True
            continue

        if claim['type'] == 'peng':
            turn = claim['claimer']
            skip_draw = True
            continue

        turn = (turn + 1) % num_agents
        msg = Message(current.name, 'put', discarded)
        for other in agents:
            if other.name == current.name:
                continue
            other.handle_msg(msg)


class _WallPool:
    """用于 play_game_from_state 的 _process_claims 兼容对象。"""
    def __init__(self, wall, idx):
        self.tiles = wall
        self.idx = idx
