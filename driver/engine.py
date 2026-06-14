# -*- coding: utf-8 -*-
"""Self-contained Mahjong game driver."""

from agent import Message
import tile_pool
from algo.eval.v2 import shanten


import time


def play_game(agents, tile_pool_cls=None, verbose=False, record_time=False,
              record_log=False):
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
    while True:
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

        current = agents[turn]
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
                'wall_remaining': wall_remaining(),
            })

        # 报听检测：未锁死、13 张、向听数为 0
        if not locked and current.name not in locked_names and len(current.cur) == 13:
            if shanten(current.cur) == 0:
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

        msg = Message(current.name, 'put', discarded)

        for i, other in enumerate(agents):
            if i == turn:
                continue
            resp = other.handle_msg(msg)
            if resp.type == 'i_win':
                result = {
                    'winner': other.name,
                    'win_type': 'ron',
                    'dealer': current.name,
                    'players_order': [a.name for a in agents],
                }
                if record_time:
                    result['decision_times'] = decision_times
                if record_log:
                    event_log.append({
                        'type': 'win',
                        'player': other.name,
                        'win_type': 'ron',
                        'tile': discarded,
                        'dealer': current.name,
                        'wall_remaining': wall_remaining(),
                    })
                    result['event_log'] = event_log
                return result

        turn = (turn + 1) % num_agents
