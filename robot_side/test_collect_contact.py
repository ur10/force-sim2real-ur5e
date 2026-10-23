"""Offline test of collect_contact.py's key handling and event output (no robot, no cameras).
Copy next to the script in ~/ring_rod_probe and run from the repo root:
    python -m pytest tests/test_collect_contact.py      (or: python tests/test_collect_contact.py)
"""
import json
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import collect_contact as cc  # noqa: E402
from utils.config import load_robot_config  # noqa: E402
from utils.events import EventLog, pair_attempts  # noqa: E402


def _collector():
    cfg = load_robot_config(None)
    d = tempfile.mkdtemp()
    log = cc.FakeLogger(500)
    ev = EventLog(os.path.join(d, 'events.jsonl'), log.latest)
    env = cc.FakeEnv(5, (1280, 720))
    live = cc.Live(log, 'actual_TCP_force', dict(bottom_below_tcp=0.0249, height=0.0157),
                   dict(axis_xy=[0.0857, -0.6381], tip_z=0.1801))
    return cc.ContactCollector(cfg, log, ev, env, live, video='attempts'), d, env


def _press(col, seq):
    for k in seq:
        col.handle(k, time.time())
        time.sleep(0.005)


def test_session_events_parse_with_ring_rod_probe():
    col, d, env = _collector()
    _press(col, ['r', '2', 'r', 'r', 'ESC', 't', 's', 's', 'p', 'p',
                 'a', 'c', 'h', 'o', '2',
                 'a', 'BACKSPACE', 'x', 'BACKSPACE', 'BACKSPACE', 'o', '1',
                 'v', 'b', 'q'])
    evs = [json.loads(line) for line in open(os.path.join(d, 'events.jsonl'))]
    types = [e['type'] for e in evs]
    atts, warns = pair_attempts(evs)
    assert [a['outcome']['result'] for a in atts] == ['jammed', 'seated'] and not warns
    assert atts[0]['outcome']['note'] == 'recovered by hand'
    assert [e['kind'] for e in evs if e['type'] == 'reg_start'] == ['tip']
    assert types.count('hold_start') == 1 and types.count('hold_end') == 1
    assert types.count('episode_end') == 2 and types.count('episode_dropped') == 1
    assert env.replay_buffer.n_episodes == 2
    assert 'offset_mm' in next(e for e in evs if e['type'] == 'attempt_start')
    assert col.stop


def test_quit_needs_confirmation_when_open():
    col, _, _ = _collector()
    _press(col, ['a', 'q'])
    assert not col.stop
    _press(col, ['q'])
    assert col.stop


def test_outcome_without_attempt_is_ignored():
    col, d, _ = _collector()
    _press(col, ['o', '1'])
    evs = [json.loads(line) for line in open(os.path.join(d, 'events.jsonl'))] if os.path.exists(os.path.join(d, 'events.jsonl')) else []
    assert not [e for e in evs if e['type'] == 'outcome']


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
