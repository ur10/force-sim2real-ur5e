#!/usr/bin/env python
"""Contact-probe and calibration collection with GELLO teleop.

collect.py's flow (GelloEnv teleop, cameras into the replay buffer, c/s/backspace/q style keys)
plus the ring_rod_probe read-only wrench log and its event marks, so every session can be
analysed with ring_rod_probe's register_rod / characterize_sensor / segment_attempts /
export_probe unchanged, and the camera episodes are linked to the attempts.

Install: copy this file to ~/ring_rod_probe/scripts/ (it imports ring_rod_probe's utils and,
unless --dry-run, the memory repo's GelloEnv).

Safety, same rules as ring_rod_probe:
  * The wrench logger is ring_rod_probe's RtdeLogger: output recipes only, no input registers,
    no motion, no settings writes. The only exception is the guarded tare with --allow-tare.
  * The robot moves only as GelloEnv moves it: env.calibrate() at start-up (asks first), then it
    follows the GELLO leader. This script sends no motion of its own and cannot stop the arm:
    above --alarm-n it warns (screen + bell + event); the UR's own protective limits stay the
    safety layer. Contact under GELLO is position-controlled (no compliance): approach slowly.
  * Start order is GelloEnv first, then the logger (GelloEnv's controller sets TCP and payload
    when it starts; the session metadata records both and any change).

Keys (no Enter needed):
  camera episodes   v  start      b  stop and save      Backspace x2  drop      q  quit (q twice if open)
  calibration       s  still segment start/end          f  free-motion segment start/end
                    r  registration capture start/end, then 1-6: seated tip side+x side-x side+y side-y
                    t  tare (pendant: mark only; --allow-tare: the guarded sensor tare in utils/tare.py)
  attempts          a  attempt start   c  contact noticed   p  press-and-hold start/end (5-10 N)
                    h  note "recovered by hand"   o  outcome, then 1 seated / 2 jammed / 3 aborted
                    i  print the coverage table   Esc  cancel a pending choice
  With --video attempts (default for --plan attempts) a and o also start and save a camera episode.

Usage:
    DISPLAY=:1 python scripts/collect_contact.py --plan sensor -l sensor1
    DISPLAY=:1 python scripts/collect_contact.py --plan reg -l reg1
    DISPLAY=:1 python scripts/collect_contact.py --plan attempts -l attempts_a --ring pla_0p3 \
        --registration outputs/rod_registration.json
    python scripts/collect_contact.py --dry-run --plan attempts -l rehearse   # no robot, no cameras
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RING_ROOT = os.path.dirname(HERE)
sys.path.insert(0, RING_ROOT)  # ring_rod_probe's utils must win over memory/utils

import datetime  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import click  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from utils.config import load_measured, load_robot_config, ring_geometry, try_geometry  # noqa: E402
from utils.events import OUTCOMES, REG_KINDS, EventLog  # noqa: E402
from utils.geometry import ring_centres, tilt_from_down_deg  # noqa: E402
from utils.session_io import save_log, write_json  # noqa: E402

PLANS = {
    'sensor': [
        'Tare once at the start (t), then not again in this session.',
        '3-5 still segments (s ... s), 10-30 s each, each at a different tool orientation;',
        '  include the attempt tilts (0, 3, 6 deg) and a few larger ones.',
        'Several free-motion segments (f ... f): slow GELLO moves near the rod without touching it,',
        '  each starting with ~0.5 s still.',
    ],
    'reg': [
        'One capture of each kind (r, pick 1-6, do it, r again):',
        '  1 seated: ring centred and seated on the rod; hold still 2 s.',
        '  2 tip (x4, from +x, -x, +y, -y): hover ~6 mm off-axis, 5 mm above the tip, open the capture,',
        '    hold >= 0.3 s, descend slowly (~2 mm/s) until the rim lands on the tip, stop, close.',
        '  3-6 side: a few mm over the rod, centred; open, hold >= 0.3 s, move ~1 mm/s in the named',
        '    direction until the bore touches the rod; stop, close.',
    ],
    'attempts': [
        'Per attempt: tare while still at the attempt tilt (t), line up 5-10 mm above the tip,',
        '  a, steady descent (~5 mm/s), c when you notice contact, o + outcome.',
        'Coverage: offsets ~0, 1, 2, 2.5, 3, 3.5, 4, 6 mm in several directions; tilts ~0, 3, 6 deg;',
        '  3+ attempts each, including deliberate off-axis jams.',
        'On some jams press to 5-10 N and hold ~1 s (p ... p): the direction signal needs it.',
        'Some jams recovered by hand (back off, re-align, retry): mark them with h before o.',
        'Keep the grasp and wrist yaw unchanged since registration.',
    ],
}
OFFSET_BINS_MM = (0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0)
TILT_BINS_DEG = (0.0, 3.0, 6.0)


def say(msg):
    print('\r\033[K' + msg, flush=True)


def load_memory_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cams = cfg['cameras']
    cams['num_cameras'] = (len(cams.get('realsense_serials') or []) + len(cams.get('flir_serials') or [])
                           + len(cams.get('extra_realsense_serials') or []))
    return cfg


def task_j_init(cfg, task):
    tasks = cfg.get('tasks', {})
    t = tasks.get(task, tasks.get('default'))
    return None if t is None else np.radians(np.asarray(t['j_init_deg'], float))


# ----------------------------------------------------------------------------- live geometry
class Live:
    """What the operator needs on screen: force, speed, and (with a registration and the
    measured grasp) the ring's offset from the rod axis, tilt and height above the tip."""

    def __init__(self, logger, channel, ring=None, reg=None):
        self.logger, self.channel, self.ring, self.reg = logger, channel, ring, reg

    def read(self):
        r = self.logger.recent(0.1)
        if r is None or len(r['timestamp']) == 0:
            return None
        w = np.asarray(r[self.channel][-1], float)
        pose = np.asarray(r['actual_TCP_pose'][-1], float)
        speed = np.asarray(r['actual_TCP_speed'][-1], float)
        t = r['timestamp']
        out = dict(F=w[:3], tau=w[3:], F_norm=float(np.linalg.norm(w[:3])), v_mm_s=float(np.linalg.norm(speed[:3]) * 1e3),
                   tilt_deg=float(tilt_from_down_deg(pose[3:])),
                   rate_hz=float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 1 and t[-1] > t[0] else 0.0)
        if self.ring is not None and self.reg is not None:
            bottom, _ = ring_centres(pose, self.ring['bottom_below_tcp'], self.ring['height'])
            off = bottom[:2] - np.asarray(self.reg['axis_xy'])
            out.update(offset_x_mm=float(off[0] * 1e3), offset_y_mm=float(off[1] * 1e3),
                       offset_mm=float(np.linalg.norm(off) * 1e3), above_tip_mm=float((bottom[2] - self.reg['tip_z']) * 1e3))
        return out


# ----------------------------------------------------------------------------- key handling
class ContactCollector:
    def __init__(self, probe_cfg, logger, events, env, live, *, allow_tare=False, video='none',
                 warn_n=10.0, alarm_n=25.0, robot_ip=None):
        self.cfg, self.logger, self.ev, self.env, self.live = probe_cfg, logger, events, env, live
        self.allow_tare, self.video, self.warn_n, self.alarm_n = allow_tare, video, warn_n, alarm_n
        self.robot_ip = robot_ip
        self.open = {'still': None, 'free': None, 'reg': None, 'hold': None}
        self.attempt = None            # open attempt event
        self.attempt_note = ''
        self.pending = None            # ('reg' | 'outcome', expires_at)
        self.recording = False
        self.auto_episode = False      # the open camera episode was started by 'a'
        self.pending_drop = False
        self.quit_armed = False
        self.coverage = {}             # (offset bin, tilt bin) -> {result: count}
        self.last_alarm = 0.0
        self.stop = False

    # -- camera episodes ---------------------------------------------------------------------
    def _episode_index(self):
        return int(self.env.replay_buffer.n_episodes)

    def start_episode(self, start_time, auto=False):
        if self.recording:
            return say('  camera episode already recording')
        self.env.start_episode(start_time)
        self.recording, self.auto_episode = True, auto
        e = self.ev.mark('episode_start', episode=self._episode_index(), auto=auto)
        say(f"[{e['t_robot'] or 0:.3f}] camera episode {self._episode_index()} recording")

    def end_episode(self):
        if not self.recording:
            return say('  no camera episode recording')
        idx = self._episode_index()
        self.env.end_episode()
        self.recording = self.auto_episode = False
        e = self.ev.mark('episode_end', episode=idx)
        say(f"[{e['t_robot'] or 0:.3f}] camera episode {idx} saved")

    def drop_episode(self):
        self.env.drop_episode()
        idx = self._episode_index()
        self.recording = self.auto_episode = False
        self.ev.mark('episode_dropped', episode=idx)
        say(f'  camera episode {idx} dropped (the wrench log keeps running)')

    # -- marks ---------------------------------------------------------------------------------
    def toggle(self, what, start, end, **extra):
        if self.open[what] is None:
            e = self.ev.mark(start, **extra)
            self.open[what] = e
            say(f"[{e['t_robot'] or 0:.3f}] {start} {extra or ''}")
        else:
            e = self.ev.mark(end)
            dur = (e['t_robot'] or 0) - (self.open[what]['t_robot'] or 0)
            self.open[what] = None
            say(f"[{e['t_robot'] or 0:.3f}] {end} ({dur:.1f} s)")
            a = self.cfg['analysis']
            if what == 'still' and not (a['still_segment_min_s'] <= dur <= a['still_segment_max_s']):
                say(f"  note: still segments should be {a['still_segment_min_s']:.0f}-{a['still_segment_max_s']:.0f} s")

    def key_r(self):
        if self.open['reg'] is not None:
            return self.toggle('reg', 'reg_start', 'reg_end')
        self.pending = ('reg', time.monotonic() + 8.0, self.logger.latest(), time.monotonic())
        say('  capture kind: ' + '  '.join(f'{i + 1} {k}' for i, k in enumerate(REG_KINDS)) + '   (Esc cancels)')

    def key_a(self, start_time):
        if self.attempt is not None:
            say('  previous attempt had no outcome; it will be flagged missing_outcome')
        cond = {}
        lv = self.live.read()
        if lv is not None:
            cond = {k: round(lv[k], 3) for k in ('offset_x_mm', 'offset_y_mm', 'offset_mm', 'above_tip_mm', 'tilt_deg') if k in lv}
        e = self.ev.mark('attempt_start', **cond)
        self.attempt, self.attempt_note = e, ''
        say(f"[{e['t_robot'] or 0:.3f}] attempt_start {cond or ''}")
        if self.video == 'attempts' and not self.recording:
            self.start_episode(start_time, auto=True)

    def key_c(self):
        e = self.ev.mark('contact')
        say(f"[{e['t_robot'] or 0:.3f}] contact" + ('' if self.attempt else '  (no attempt open)'))

    def key_o(self):
        if self.attempt is None:
            return say('  no attempt open (press a first); outcome not recorded')
        self.pending = ('outcome', time.monotonic() + 8.0, self.logger.latest(), time.monotonic())
        say('  outcome: 1 seated  2 jammed  3 aborted   (Esc cancels)')

    def choose(self, digit):
        kind, _, sample, t_key = self.pending
        self.pending = None
        if kind == 'reg':
            if not 1 <= digit <= len(REG_KINDS):
                return say('  cancelled')
            k = REG_KINDS[digit - 1]
            e = self.ev.mark('reg_start', sample=sample, t_host=t_key, kind=k)
            self.open['reg'] = e
            return say(f"[{e['t_robot'] or 0:.3f}] reg_start kind={k}  (r again to end)")
        if not 1 <= digit <= len(OUTCOMES):
            return say('  cancelled; attempt still open')
        result = OUTCOMES[digit - 1]
        e = self.ev.mark('outcome', sample=sample, t_host=t_key, result=result, note=self.attempt_note)
        self._count(self.attempt, result)
        self.attempt = None
        say(f"[{e['t_robot'] or 0:.3f}] outcome {result}" + (f' ({self.attempt_note})' if self.attempt_note else ''))
        if self.recording and self.auto_episode:
            self.end_episode()

    def key_t(self):
        if not self.allow_tare:
            e = self.ev.mark('tare', method='pendant')
            return say(f"[{e['t_robot'] or 0:.3f}] tare marked (pendant). Nothing was sent to the robot.")
        from utils.tare import check_tare_effect, send_tare, tare_gate
        tc = self.cfg['tare']
        ok, reason = tare_gate(self.logger.recent(tc['min_still_s'] + 0.2), time.monotonic(), self.logger.frequency,
                               tc['max_tcp_speed_mm_s'], tc['min_still_s'], tc['max_tcp_ang_speed_deg_s'], tc['max_data_age_s'])
        if not ok:
            self.ev.mark('tare_refused', reason=reason)
            return say(f'  TARE REFUSED: {reason}')
        before = self.logger.recent(0.2)['actual_TCP_force']
        sample = self.logger.latest()
        try:
            send_tare(self.robot_ip)
        except OSError as exc:
            self.ev.mark('tare_failed', error=repr(exc))
            return say(f'  tare send FAILED: {exc!r}')
        e = self.ev.mark('tare', sample=sample, method='script', gate=reason)
        time.sleep(0.4)
        res = check_tare_effect(before, self.logger.recent(0.15)['actual_TCP_force'])
        self.ev.mark('tare_check', tare_t_robot=e['t_robot'], **res)
        say(f"[{e['t_robot'] or 0:.3f}] tare {'confirmed' if res['confirmed'] else res['confirmed']}  "
            f"|F| {res['force_before_n']:.2f} -> {res['force_after_n']:.2f} N")

    def _count(self, start_ev, result):
        if start_ev is None or 'offset_mm' not in start_ev:
            return
        ob = min(OFFSET_BINS_MM, key=lambda b: abs(b - start_ev['offset_mm']))
        tb = min(TILT_BINS_DEG, key=lambda b: abs(b - start_ev.get('tilt_deg', 0.0)))
        self.coverage.setdefault((ob, tb), {}).setdefault(result, 0)
        self.coverage[(ob, tb)][result] += 1

    def coverage_table(self):
        lines = ['coverage (attempts with a registration; offset bin x tilt bin -> outcomes)']
        hdr = 'offset mm | ' + ' | '.join(f'tilt {t:g}' for t in TILT_BINS_DEG)
        lines.append(hdr)
        for o in OFFSET_BINS_MM:
            cells = []
            for t in TILT_BINS_DEG:
                c = self.coverage.get((o, t), {})
                cells.append(f"{sum(c.values()):2d} ({c.get('seated', 0)}s/{c.get('jammed', 0)}j)")
            lines.append(f'{o:9g} | ' + ' | '.join(cells))
        return '\n'.join(lines)

    def key_q(self):
        busy = [k for k, v in self.open.items() if v is not None] + (['attempt'] if self.attempt else []) \
            + (['camera episode'] if self.recording else [])
        if busy and not self.quit_armed:
            self.quit_armed = True
            return say(f"  {', '.join(busy)} still open: press q again to quit anyway")
        if self.recording:
            self.end_episode()
        self.ev.mark('quit')
        self.stop = True

    def handle(self, key, start_time):
        """key: a single character, or 'BACKSPACE' / 'ESC' / 'SPACE'."""
        if key != 'q':
            self.quit_armed = False
        if key != 'BACKSPACE' and self.pending_drop:
            self.pending_drop = False
            say('  drop cancelled')
        if self.pending is not None:
            if time.monotonic() > self.pending[1] or key == 'ESC':
                self.pending = None
                return say('  choice cancelled')
            if key.isdigit():
                return self.choose(int(key))
            self.pending = None
            say('  choice cancelled')
        if key == 'BACKSPACE':
            if not self.recording:
                return say('  no camera episode recording')
            if not self.pending_drop:
                self.pending_drop = True
                return say('  press Backspace again to drop the camera episode, any other key to cancel')
            self.pending_drop = False
            return self.drop_episode()
        table = {'v': lambda: self.start_episode(start_time), 'b': self.end_episode, 'q': self.key_q,
                 's': lambda: self.toggle('still', 'still_start', 'still_end'),
                 'f': lambda: self.toggle('free', 'free_start', 'free_end'),
                 'p': lambda: self.toggle('hold', 'hold_start', 'hold_end'),
                 'r': self.key_r, 'a': lambda: self.key_a(start_time), 'c': self.key_c, 'o': self.key_o,
                 't': self.key_t, 'i': lambda: print('\n' + self.coverage_table()),
                 'h': self._note_recovered}
        if key in table:
            table[key]()

    def _note_recovered(self):
        if self.attempt is None:
            return say('  no attempt open')
        self.attempt_note = 'recovered by hand'
        self.ev.mark('note', text=self.attempt_note)
        say('  note: recovered by hand (saved with the outcome)')

    def watch_force(self, lv):
        if lv is None or lv['F_norm'] < self.alarm_n:
            return
        now = time.monotonic()
        if now - self.last_alarm > 1.0:
            self.last_alarm = now
            self.ev.mark('force_alarm', F_n=round(lv['F_norm'], 2))
            print('\a', end='')
            say(f"  FORCE {lv['F_norm']:.1f} N > {self.alarm_n:.0f} N: back off on the GELLO")

    def status(self, lv):
        opened = [k for k, e in self.open.items() if e is not None] + (['attempt'] if self.attempt else [])
        if lv is None:
            return 'waiting for wrench data'
        s = f"{lv['rate_hz']:4.0f} Hz | |F| {lv['F_norm']:5.2f} N  Fz {lv['F'][2]:+6.2f} | v {lv['v_mm_s']:5.1f} mm/s | tilt {lv['tilt_deg']:4.1f}"
        if 'offset_mm' in lv:
            s += f" | off {lv['offset_x_mm']:+5.2f},{lv['offset_y_mm']:+5.2f} mm | above tip {lv['above_tip_mm']:+6.2f} mm"
        return s + f" | open: {','.join(opened) or '-'}" + (' | REC' if self.recording else '')


def overlay(img, collector, lv):
    import cv2
    h = img.shape[0]
    lines = [collector.status(lv)[i:i + 95] for i in range(0, len(collector.status(lv)), 95)]
    col = (255, 255, 255)
    if lv is not None and lv['F_norm'] >= collector.alarm_n:
        col = (0, 0, 255)
    elif lv is not None and lv['F_norm'] >= collector.warn_n:
        col = (0, 200, 255)
    for k, line in enumerate(lines):
        cv2.putText(img, line, (10, h - 40 - 24 * (len(lines) - 1 - k)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, line, (10, h - 40 - 24 * (len(lines) - 1 - k)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)
    if lv is not None and lv['F_norm'] >= collector.alarm_n:
        cv2.putText(img, f"FORCE {lv['F_norm']:.0f} N - BACK OFF", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return img


# ----------------------------------------------------------------------------- dry run fakes
class _FakeBuffer:
    n_episodes = 0


class FakeEnv:
    """No robot, no cameras: blank frames; episode calls only counted."""

    def __init__(self, num_cameras, res):
        self.replay_buffer, self.num_cameras, self.res = _FakeBuffer(), num_cameras, res

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def calibrate(self):
        pass

    def get_obs(self):
        return {f'camera{i}_rgb': np.zeros((1, self.res[1], self.res[0], 3), np.uint8) for i in range(self.num_cameras)}

    def start_episode(self, t):
        pass

    def end_episode(self):
        self.replay_buffer.n_episodes += 1

    def drop_episode(self):
        pass

    def exec_actions(self, actions, timestamps):
        pass


class FakeLogger:
    """Synthetic 500 Hz samples: the TCP descends slowly toward a tip, with force noise."""

    def __init__(self, frequency=500.0):
        self.frequency, self.t0, self.idx = frequency, time.monotonic(), 0
        self.alive, self.error = True, None

    def _rows(self, seconds):
        now = time.monotonic() - self.t0
        t = np.arange(max(0.0, now - seconds), now, 1 / self.frequency)
        if len(t) == 0:
            t = np.array([now])
        z = 0.20 - 0.002 * (t % 10)
        pose = np.stack([np.full_like(t, 0.0857), np.full_like(t, -0.6381), z, np.full_like(t, math.pi), 0 * t, 0 * t], -1)
        rng = np.random.default_rng(int(now * 1000))
        w = rng.normal(0, 0.2, (len(t), 6))
        speed = np.zeros((len(t), 6))
        speed[:, 2] = -0.002
        return dict(timestamp=t, actual_TCP_pose=pose, actual_TCP_speed=speed, actual_TCP_force=w, ft_raw_wrench=w)

    def start(self):
        pass

    def latest(self):
        r = self._rows(0.002)
        self.idx += 1
        return {'timestamp': float(r['timestamp'][-1]), 'idx': self.idx}

    def recent(self, seconds=None):
        return self._rows(seconds or 1.0)

    def stop(self):
        return {'timestamp': np.zeros(0)}, {'frequency': self.frequency, 'dry_run': True}

    def cleanup_chunks(self):
        pass


# ----------------------------------------------------------------------------- main
@click.command()
@click.option('--plan', type=click.Choice(sorted(PLANS)), required=True, help='checklist and defaults for this session')
@click.option('--label', '-l', required=True, help='short session name, e.g. sensor1, reg1, attempts_a')
@click.option('--ring', default='main', help='ring in use: main or a printed_rings name in measured.json')
@click.option('--task', '-t', default='ring_rod', help='memory config.yaml task for j_init (falls back to default)')
@click.option('--memory-root', default=os.path.expanduser('~/memory'), help='memory repo (GelloEnv, config.yaml, viewer)')
@click.option('--memory-config', default=None, help='memory config.yaml (default: <memory-root>/config.yaml)')
@click.option('--config', '-c', default=None, help='ring_rod_probe robot.yaml (default: config/robot.yaml)')
@click.option('--measured', '-m', default=None, help='ring_rod_probe measured.json (default: config/measured.json)')
@click.option('--registration', default=None, help='rod_registration.json: enables the live offset / height display')
@click.option('--output_dir', '-o', default='data/sessions', help='wrench sessions (ring_rod_probe layout)')
@click.option('--video-dir', default=os.path.expanduser('~/Desktop/data/'), help='camera replay buffers (collect.py layout)')
@click.option('--video', type=click.Choice(['default', 'none', 'attempts', 'manual']), default='default',
              help='attempts: a/o start and save a camera episode; manual: only v/b; none: never')
@click.option('--allow-tare', is_flag=True, help="t sends ring_rod_probe's guarded sensor tare, utils/tare.py (needs Remote Control)")
@click.option('--warn-n', default=10.0, help='force shown in orange above this (N)')
@click.option('--alarm-n', default=25.0, help='force alarm above this (N): screen, bell, event')
@click.option('--yes', is_flag=True, help='skip the confirmation before env.calibrate() moves the arm')
@click.option('--dry-run', is_flag=True, help='no robot, no cameras, synthetic wrench: rehearse the keys')
def main(plan, label, ring, task, memory_root, memory_config, config, measured, registration, output_dir, video_dir,
         video, allow_tare, warn_n, alarm_n, yes, dry_run):
    probe_cfg = load_robot_config(config)
    mem_cfg = load_memory_config(memory_config or os.path.join(memory_root, 'config.yaml'))
    if mem_cfg['robot']['ip'] != probe_cfg['robot']['ip']:
        raise click.UsageError(f"robot IP differs: memory {mem_cfg['robot']['ip']} vs ring_rod_probe {probe_cfg['robot']['ip']}")
    video = {'attempts': 'attempts', 'sensor': 'none', 'reg': 'none'}[plan] if video == 'default' else video
    meas = load_measured(measured)
    ring_geo, why = try_geometry(ring_geometry, meas, ring, need_radii=False)
    reg = None
    if registration:
        with open(registration) as f:
            reg = json.load(f)
    if ring != 'main' and ring_geo is None:
        raise click.BadParameter(why, param_hint='--ring')

    sdir = os.path.join(output_dir, f'{datetime.datetime.now():%Y%m%d_%H%M%S}_{label}')
    vdir = os.path.join(video_dir, f'ring_rod_{label}')
    cams = mem_cfg['cameras']
    res = tuple(cams['obs_resolution'])
    freq = mem_cfg['robot']['frequency']
    latency = mem_cfg['robot']['command_latency']
    dt = 1 / freq
    j_init = task_j_init(mem_cfg, task)

    print('\n'.join([f'Plan: {plan}'] + ['  ' + s for s in PLANS[plan]]))
    print(__doc__.split('Keys (no Enter needed):')[1].split('Usage:')[0])
    print(f"wrench session -> {sdir}\ncamera episodes ({video}) -> {vdir}\nring: {ring}  "
          f"tare: {'guarded script' if allow_tare else 'pendant, mark only'}  "
          f"live offset: {'on' if (ring_geo and reg) else 'off (' + (why or 'no --registration') + ')'}")

    viewer = None
    if dry_run:
        from utils.keys import TerminalKeys
        env_cm, keys_cm = FakeEnv(cams['num_cameras'], res), TerminalKeys()
        logger = FakeLogger(probe_cfg['rtde']['frequency'])
    else:
        sys.path.append(memory_root)                          # appended: ring_rod_probe utils stay first
        sys.path.append(os.path.join(memory_root, 'real_robot'))
        from multiprocessing.managers import SharedMemoryManager
        from real_robot.robot_controller.gello_env import GelloEnv
        from real_robot.robot_controller.keystroke_counter import KeystrokeCounter
        from utils.rtde_logger import RtdeLogger
        viewer_script = os.path.join(memory_root, 'scripts', 'viewer.py')
        if os.path.exists(viewer_script):
            viewer = subprocess.Popen([sys.executable, viewer_script], env={**os.environ, 'DISPLAY': os.environ.get('DISPLAY', ':1')})
        shm = SharedMemoryManager()
        shm.start()  # shut down in finally, like collect.py's context manager
        env_cm = GelloEnv(output_dir=vdir, robot_ip=mem_cfg['robot']['ip'], obs_image_resolution=res, frequency=freq,
                          enable_multi_cam_vis=False, shm_manager=shm, j_init=j_init,
                          flir_camera_serial_numbers=cams['flir_serials'], rs_camera_serial_numbers=cams['realsense_serials'],
                          rs_extra_camera_serial_numbers=cams.get('extra_realsense_serials') or [])
        keys_cm = KeystrokeCounter()
        logger = None
    try:
        with keys_cm as keys, env_cm as env:
            if not dry_run:
                if not yes and input('env.calibrate() moves the UR5e to the GELLO pose. Clear the cell and type y: ').strip() != 'y':
                    print('aborted before calibrate')
                    return
            env.calibrate()
            if not dry_run:
                keys.clear()  # drop the keys typed at the prompt
            if not dry_run:  # the logger starts after the teleop controller has set TCP / payload
                logger = RtdeLogger(probe_cfg['robot']['ip'], probe_cfg['rtde']['frequency'], probe_cfg['rtde']['fallback_frequency'],
                                    out_dir=sdir, timeout=probe_cfg['rtde']['connect_timeout_s'])
            os.makedirs(sdir, exist_ok=True)
            logger.start()
            ev = EventLog(os.path.join(sdir, 'events.jsonl'), logger.latest)
            info = dict(label=label, ring=ring, plan=plan, video=video, video_dir=vdir, tare_mode='script' if allow_tare else 'pendant',
                        robot_config=probe_cfg, memory_config={k: mem_cfg[k] for k in ('robot', 'cameras')}, task=task,
                        j_init_rad=None if j_init is None else j_init.tolist(), registration=registration, argv=sys.argv,
                        dry_run=dry_run, script='collect_contact.py')
            write_json(os.path.join(sdir, 'session_info.json'), info)
            live = Live(logger, probe_cfg['wrench']['channel'], ring_geo, reg)
            col = ContactCollector(probe_cfg, logger, ev, env, live, allow_tare=allow_tare, video=video, warn_n=warn_n,
                                   alarm_n=alarm_n, robot_ip=probe_cfg['robot']['ip'])
            time.sleep(1.0 if dry_run else 2.0)
            t_start, it, last_status = time.monotonic(), 0, 0.0
            while not col.stop:
                t_cycle_end = t_start + (it + 1) * dt
                t_sample = t_cycle_end - latency
                t_command = t_cycle_end + dt
                obs = env.get_obs()
                start_time = t_start + (it + 2) * dt - time.monotonic() + time.time()
                for k in _keys(keys, dry_run):
                    col.handle(k, start_time)
                    if not dry_run:
                        keys.clear()
                lv = live.read()
                col.watch_force(lv)
                if not dry_run:
                    _show(obs, cams['num_cameras'], res, col, lv, env)
                if time.monotonic() - last_status > 0.5:
                    print('\r\033[K' + col.status(lv), end='', flush=True)
                    last_status = time.monotonic()
                if not logger.alive:
                    say(f'LOGGER STOPPED: {logger.error}. Saving what was recorded.')
                    break
                _wait_until(t_sample)
                env.exec_actions(actions=[[0, 0, 0, 0, 0, 0, 0]], timestamps=[t_command - time.monotonic() + time.time()])
                _wait_until(t_cycle_end)
                it += 1
    finally:
        print()
        if viewer is not None:
            viewer.terminate()
        if not dry_run and 'shm' in locals():
            shm.shutdown()
        if os.path.exists('/tmp/live_frame.npy') and not dry_run:
            os.remove('/tmp/live_frame.npy')
        if logger is not None:
            arrays, meta = logger.stop()
            meta.update(label=label, ring=ring, plan=plan, video_dir=vdir, dry_run=dry_run)
            events = ev.events if 'ev' in locals() else []
            save_log(os.path.join(sdir, 'session.npz'), arrays, meta, events)
            logger.cleanup_chunks()
            if not dry_run:
                from utils.report import log_report
                print(log_report(meta))
            print(f'{len(events)} marks -> {sdir}/session.npz')
            print(_next_steps(plan, sdir))


def _keys(keys, dry_run):
    if dry_run:
        k = keys.get(0.0)
        if k is None:
            return []
        return [{'\x7f': 'BACKSPACE', '\x08': 'BACKSPACE', '\x1b': 'ESC', ' ': 'SPACE'}.get(k, k)]
    from real_robot.robot_controller.keystroke_counter import Key, KeyCode
    out = []
    for ks in keys.get_press_events():
        if isinstance(ks, KeyCode) and ks.char:
            out.append(ks.char)
        elif ks == Key.backspace:
            out.append('BACKSPACE')
        elif ks == Key.esc:
            out.append('ESC')
        elif ks == Key.space:
            out.append('SPACE')
    return out


def _wait_until(t):
    try:
        from real_robot.robot_controller.common.precise_sleep import precise_wait
        precise_wait(t)
    except ImportError:
        time.sleep(max(0.0, t - time.monotonic()))


def _show(obs, n, res, col, lv, env):
    """collect.py's grid of camera views, with the wrench status drawn on it, to the viewer."""
    import cv2
    ims = []
    for i in range(n):
        try:
            ims.append(obs[f'camera{i}_rgb'][-1, :, :, ::-1].copy())
        except Exception:
            ims.append(np.zeros((res[1], res[0], 3), np.uint8))
    n_cols = 2 if len(ims) <= 4 else 3
    while len(ims) % n_cols or len(ims) < 4:
        ims.append(np.zeros_like(ims[0]))
    grid = np.vstack([np.hstack(ims[r:r + n_cols]) for r in range(0, len(ims), n_cols)])
    s = 1280 / grid.shape[1]
    if s < 1:
        grid = cv2.resize(grid, None, fx=s, fy=s)
    if col.recording:
        cv2.rectangle(grid, (0, 0), (grid.shape[1] - 1, grid.shape[0] - 1), (0, 0, 255), 3)
    overlay(grid, col, lv)
    cv2.putText(grid, f'Episode: {env.replay_buffer.n_episodes}' + (', Recording!' if col.recording else ''),
                (10, grid.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    np.save('/tmp/live_frame.npy', grid)


def _next_steps(plan, sdir):
    return {
        'sensor': f'next: python scripts/characterize_sensor.py {sdir} -o outputs/sensor',
        'reg': f'next: python scripts/register_rod.py {sdir} -o outputs/rod_registration.json',
        'attempts': (f'next: python scripts/segment_attempts.py {sdir} -r outputs/rod_registration.json -o outputs/attempts.json\n'
                     f'      python scripts/export_probe.py -a outputs/attempts.json -o outputs/export'),
    }[plan]


if __name__ == '__main__':
    main()
