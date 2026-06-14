#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mahjong game replay GUI."""

import sys
import os
import json
import argparse
import tkinter as tk
from tkinter import ttk, messagebox

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tile


# 花色颜色
SUIT_COLORS = {
    0: '#c0392b',  # 万
    1: '#27ae60',  # 条
    2: '#2980b9',  # 筒
    3: '#2c3e50',  # 字
}


def _suit_of(t):
    return t // 10 if t < 30 else 3


def _tile_color(t):
    return SUIT_COLORS.get(_suit_of(t), '#000000')


class ReplayApp:
    def __init__(self, root, event_log):
        self.root = root
        self.event_log = event_log
        self.idx = 0

        self.root.title('Mahjong AI Replay')
        self.root.geometry('1100x750')

        self._build_ui()
        self._render()
        self._bind_keys()

    def _build_ui(self):
        # 顶部状态栏
        self.status_var = tk.StringVar(value='')
        self.event_var = tk.StringVar(value='')
        status_bar = tk.Frame(self.root, bg='#f0f0f0', padx=10, pady=8)
        status_bar.pack(fill=tk.X)
        tk.Label(status_bar, textvariable=self.status_var,
                 font=('Helvetica', 12, 'bold'), bg='#f0f0f0').pack(side=tk.LEFT)
        tk.Label(status_bar, textvariable=self.event_var,
                 font=('Helvetica', 11), bg='#f0f0f0', fg='#555').pack(side=tk.RIGHT)

        # 玩家面板容器
        self.players_frame = tk.Frame(self.root, padx=10, pady=10)
        self.players_frame.pack(fill=tk.BOTH, expand=True)

        init_event = self.event_log[0]
        self.players = init_event['players']
        self.player_frames = {}
        self.hand_texts = {}
        self.discard_texts = {}
        self.name_vars = {}

        # 强制 2x2 四格等宽等高，避免内容变化时 panel 宽度抖动
        for c in (0, 1):
            self.players_frame.grid_columnconfigure(
                c, weight=1, uniform='panel_col', minsize=480)
        for r in (0, 1):
            self.players_frame.grid_rowconfigure(
                r, weight=1, uniform='panel_row', minsize=300)

        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for player, (r, c) in zip(self.players, positions):
            frame = tk.LabelFrame(self.players_frame, text='', padx=10, pady=10,
                                  font=('Helvetica', 12, 'bold'), width=500, height=320)
            frame.grid(row=r, column=c, padx=10, pady=10, sticky='nsew')
            frame.grid_propagate(False)
            self.player_frames[player] = frame

            name_var = tk.StringVar(value=player)
            self.name_vars[player] = name_var
            # 固定宽度，防止 [报听]/<-- 导致 frame 请求尺寸变化
            tk.Label(frame, textvariable=name_var,
                     font=('Helvetica', 13, 'bold'), anchor='w', width=24
                     ).pack(fill=tk.X)

            tk.Label(frame, text='手牌:', font=('Helvetica', 10),
                     anchor='w').pack(fill=tk.X, pady=(8, 2))
            hand_txt = tk.Text(frame, width=28, height=2,
                               font=('Helvetica', 13, 'bold'),
                               wrap=tk.WORD, padx=4, pady=4,
                               relief=tk.RIDGE, borderwidth=1)
            hand_txt.pack(fill=tk.X)
            self._config_tile_tags(hand_txt)
            self.hand_texts[player] = hand_txt

            tk.Label(frame, text='弃牌:', font=('Helvetica', 10),
                     anchor='w').pack(fill=tk.X, pady=(10, 2))
            discard_txt = tk.Text(frame, width=28, height=4,
                                  font=('Helvetica', 13, 'bold'),
                                  wrap=tk.WORD, padx=4, pady=4,
                                  relief=tk.RIDGE, borderwidth=1)
            discard_txt.pack(fill=tk.X)
            self._config_tile_tags(discard_txt)
            self.discard_texts[player] = discard_txt

        # 控制栏
        ctrl = tk.Frame(self.root, pady=10)
        ctrl.pack(fill=tk.X)
        tk.Button(ctrl, text='|<< First', command=self._first).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text='< Prev', command=self._prev).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text='Next >', command=self._next).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text='Last >>|', command=self._last).pack(side=tk.LEFT, padx=5)

        hint = tk.Label(ctrl, text='快捷键: ← 上一步   → 下一步   Home 首步   End 末步',
                        font=('Helvetica', 10), fg='#666')
        hint.pack(side=tk.RIGHT, padx=15)

    def _bind_keys(self):
        self.root.bind('<Left>', lambda e: self._prev())
        self.root.bind('<Right>', lambda e: self._next())
        self.root.bind('<Home>', lambda e: self._first())
        self.root.bind('<End>', lambda e: self._last())

    def _state_at(self, idx):
        """根据事件日志前 idx 项重建状态。"""
        init_event = self.event_log[0]
        hands = {p: list(init_event['hands'][p]) for p in self.players}
        discards = {p: [] for p in self.players}
        tenpai = set()
        wall_remaining = init_event.get('wall_remaining', 0)
        current_player = None
        current_event = None

        for i in range(1, idx + 1):
            ev = self.event_log[i]
            current_event = ev
            wall_remaining = ev.get('wall_remaining', wall_remaining)
            player = ev.get('player')
            if player is not None:
                current_player = player
            t = ev.get('tile')
            typ = ev['type']
            if typ == 'draw' and t is not None:
                hands[player].append(t)
            elif typ == 'discard' and t is not None:
                if t in hands[player]:
                    hands[player].remove(t)
                discards[player].append(t)
            elif typ == 'tenpai':
                tenpai.add(player)
            elif typ == 'win':
                # 胡牌时若已记录 draw/discard，状态已经正确；
                # 荣和时赢家手牌中还没有这张牌，可加上以便显示。
                if ev.get('win_type') == 'ron':
                    hands[player].append(t)
        return {
            'hands': hands,
            'discards': discards,
            'tenpai': tenpai,
            'wall_remaining': wall_remaining,
            'current_player': current_player,
            'event': current_event,
            'event_index': idx,
        }

    def _render(self):
        state = self._state_at(self.idx)
        total = len(self.event_log) - 1
        ev = state['event']

        # 状态栏
        self.status_var.set(
            'Step {}/{}  |  Wall remaining: {}  |  Current: {}'.format(
                state['event_index'], total,
                state['wall_remaining'],
                state['current_player'] or '-'))
        self.event_var.set(self._describe_event(ev))

        for player in self.players:
            frame = self.player_frames[player]
            name_text = player
            if player in state['tenpai']:
                name_text += '  [报听]'
            if player == state['current_player']:
                name_text += '  <--'
                frame.config(fg='#c0392b')
            else:
                frame.config(fg='#000000')
            self.name_vars[player].set(name_text)

            self._render_tiles(self.hand_texts[player], sorted(state['hands'][player]))
            self._render_tiles(self.discard_texts[player], state['discards'][player])

    def _config_tile_tags(self, text_widget):
        for suit, color in SUIT_COLORS.items():
            text_widget.tag_config(f'suit{suit}', foreground=color)

    def _render_tiles(self, text_widget, tiles):
        text_widget.config(state=tk.NORMAL)
        text_widget.delete('1.0', tk.END)
        for i, t in enumerate(tiles):
            if i > 0:
                text_widget.insert(tk.END, ' ')
            tag = f'suit{_suit_of(t)}'
            text_widget.insert(tk.END, tile.tile_to_str(t), tag)
        text_widget.config(state=tk.DISABLED)

    def _describe_event(self, ev):
        if ev is None:
            return '初始状态'
        typ = ev['type']
        if typ == 'init':
            return '初始发牌'
        if typ == 'draw':
            return '{} 摸 {}'.format(ev['player'], tile.tile_to_str(ev['tile']))
        if typ == 'discard':
            locked = '（报听锁定）' if ev.get('locked') else ''
            return '{} 打 {}{}'.format(ev['player'], tile.tile_to_str(ev['tile']), locked)
        if typ == 'tenpai':
            return '{} 报听'.format(ev['player'])
        if typ == 'win':
            wt = '自摸' if ev.get('win_type') == 'self' else '点和'
            dealer = ev.get('dealer')
            dealer_text = ' 点炮者: {}'.format(dealer) if dealer else ''
            return '{} {} {}{}'.format(ev['player'], wt, tile.tile_to_str(ev['tile']), dealer_text)
        if typ == 'draw_end':
            return '流局'
        return str(ev)

    def _first(self):
        self.idx = 0
        self._render()

    def _last(self):
        self.idx = len(self.event_log) - 1
        self._render()

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._render()

    def _next(self):
        if self.idx < len(self.event_log) - 1:
            self.idx += 1
            self._render()


def main():
    parser = argparse.ArgumentParser(description='Replay a recorded mahjong game.')
    parser.add_argument('log', help='Path to replay JSON file.')
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print('File not found:', args.log)
        sys.exit(1)

    with open(args.log, 'r', encoding='utf-8') as f:
        data = json.load(f)

    event_log = data.get('event_log')
    if not event_log:
        print('No event_log in file.')
        sys.exit(1)

    root = tk.Tk()
    app = ReplayApp(root, event_log)
    root.mainloop()


if __name__ == '__main__':
    main()
