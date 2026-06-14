#!/usr/bin/env python3
"""Headless-ish smoke test for replay_gui: load a log and exercise navigation."""
import json
import sys
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from replay_gui import ReplayApp


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/test_replay_gui.py <replay.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    events = data['event_log']
    root = tk.Tk()
    app = ReplayApp(root, events)
    print(f"Loaded {len(events)} events")

    def step(i):
        if i >= len(events) + 6:
            print(f"Final step idx: {app.idx}")
            assert 0 <= app.idx < len(events), f"idx {app.idx} out of range"
            print("Smoke test passed")
            root.destroy()
            return
        # Alternate navigation methods; finish with _last to reach end
        actions = [app._next, app._next, app._prev, app._first, app._last, app._prev]
        actions[i % 6]()
        if i % 20 == 0:
            print(f"  step {i}: idx={app.idx}")
        root.after(20, lambda: step(i + 1))

    root.after(200, lambda: step(0))
    root.mainloop()


if __name__ == '__main__':
    main()
