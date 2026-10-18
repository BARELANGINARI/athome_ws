"""Utilitas baca + validasi poses.yaml (dipakai check_poses.py & test_poses.py).

Tidak butuh ROS (murni Python + PyYAML), jadi bisa juga dipanggil langsung:
  python3 src/spawn/scripts/check_poses.py
Validasi memakai peta navigation (map.yaml + map.pgm) bila tersedia untuk
memastikan preset tidak jatuh di dalam dinding / di luar peta.
"""
import os

import yaml

# Radius (m) sekitar titik preset yang dicek okupansinya.
CHECK_RADIUS_M = 0.15


def load_presets(poses_file):
    """-> (presets: dict, error: str|None)."""
    if not os.path.isfile(poses_file):
        return {}, f'file tidak ada: {poses_file}'
    try:
        with open(poses_file, 'r') as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        return {}, f'YAML rusak di {poses_file}: {e}'
    if not isinstance(data, dict) or not isinstance(data.get('presets'), dict) \
            or not data.get('presets'):
        return {}, f'isi {poses_file} tidak punya blok "presets: {...}" yang tidak kosong'
    return data['presets'], None


def _read_pgm(path):
    """Parser PGM (P5 biner / P2 ascii) minim tanpa dependensi -> (w, h, bytes)."""
    with open(path, 'rb') as f:
        data = f.read()
    toks, i = [], 0
    while len(toks) < 4 and i < len(data):
        c = data[i:i + 1]
        if c.isspace():
            i += 1
        elif c == b'#':
            while i < len(data) and data[i:i + 1] != b'\n':
                i += 1
        else:
            j = i
            while j < len(data) and not data[j:j + 1].isspace():
                j += 1
            toks.append(data[i:j])
            i = j
    if len(toks) < 4:
        raise ValueError(f'header PGM tidak lengkap: {path}')
    magic, w, h = toks[0], int(toks[1]), int(toks[2])
    i += 1  # satu whitespace setelah maxval
    body = data[i:]
    if magic == b'P5':
        return w, h, bytes(body[:w * h])
    if magic == b'P2':
        vals = bytes(int(v) for v in body.split()[:w * h])
        return w, h, vals
    raise ValueError(f'format PGM tidak didukung ({magic!r}) di {path}')


def load_map(map_yaml_path):
    """-> dict info peta, atau None bila file tidak ada / tidak terbaca."""
    if not map_yaml_path or not os.path.isfile(map_yaml_path):
        return None
    try:
        with open(map_yaml_path, 'r') as f:
            meta = yaml.safe_load(f) or {}
        image = meta['image']
        if not os.path.isabs(image):
            image = os.path.join(os.path.dirname(os.path.abspath(map_yaml_path)), image)
        w, h, body = _read_pgm(image)
        ox, oy, _ = meta.get('origin', [0.0, 0.0, 0.0])
        return {
            'path': map_yaml_path, 'w': w, 'h': h, 'body': body,
            'res': float(meta['resolution']),
            'ox': float(ox), 'oy': float(oy),
            'occupied_thresh': float(meta.get('occupied_thresh', 0.65)),
            'free_thresh': float(meta.get('free_thresh', 0.196)),
        }
    except (OSError, KeyError, ValueError, TypeError) as e:
        print(f'[WARN] peta tidak terbaca ({map_yaml_path}): {e}')
        return None


def _cell_state(map_info, x, y):
    """'free' | 'occupied' | 'unknown' | 'out' untuk koordinat dunia (x, y)."""
    res, h = map_info['res'], map_info['h']
    mx = int((x - map_info['ox']) / res)
    my = int(h - (y - map_info['oy']) / res)  # baris pgm 0 = atas = y terbesar
    if not (0 <= mx < map_info['w'] and 0 <= my < h):
        return 'out'
    v = map_info['body'][my * map_info['w'] + mx]
    p = (255.0 - v) / 255.0  # negate: 0 (putih = bebas)
    if p > map_info['occupied_thresh']:
        return 'occupied'
    if p < map_info['free_thresh']:
        return 'free'
    return 'unknown'


def check_entry(entry, map_info=None):
    """-> (status: 'OK'|'WARN'|'FAIL', x, y, yaw_deg, catatan: list[str])."""
    notes = []
    if not isinstance(entry, dict):
        return 'FAIL', None, None, None, ['bukan mapping {x, y, yaw_deg}']
    try:
        x = float(entry.get('x'))
        y = float(entry.get('y'))
    except (TypeError, ValueError):
        return 'FAIL', None, None, None, [f"x/y bukan angka: {entry.get('x')!r}, {entry.get('y')!r}"]
    try:
        yaw = float(entry.get('yaw_deg', 0.0))
    except (TypeError, ValueError):
        yaw = 0.0
        notes.append(f"yaw_deg bukan angka ({entry.get('yaw_deg')!r}) -> dipakai 0")
    status = 'WARN' if notes else 'OK'

    if map_info:
        res = map_info['res']
        cells = []
        r = int(CHECK_RADIUS_M / res)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                cells.append(_cell_state(map_info, x + dx * res, y + dy * res))
        if 'out' in cells:
            return 'FAIL', x, y, yaw, notes + ['di luar area peta']
        n_occ = cells.count('occupied')
        n_unk = cells.count('unknown')
        if n_occ:
            note = f'mentok dinding ({n_occ} sel sekitar terisi, radius {CHECK_RADIUS_M} m)'
            notes.append(note)
            status = 'WARN' if status == 'OK' else status
        elif n_unk:
            notes.append('area belum dipetakan (abu-abu)')
            status = 'WARN' if status == 'OK' else status
        else:
            notes.append('area bebas')
    return status, x, y, yaw, notes


def evaluate(presets, map_info=None):
    """-> (rows: list[dict], counts: dict)."""
    rows, counts = [], {'OK': 0, 'WARN': 0, 'FAIL': 0}
    seen = {}
    for name, entry in presets.items():
        status, x, y, yaw, notes = check_entry(entry, map_info)
        if x is not None:
            key = (round(x, 3), round(y, 3))
            if key in seen:
                # Bukan error: alias/duplikat sengaja boleh, cukup diinformasikan.
                notes.append(f'koordinat sama dengan preset "{seen[key]}"')
            else:
                seen[key] = str(name)
        counts[status] += 1
        rows.append({'name': str(name), 'status': status,
                     'x': x, 'y': y, 'yaw': yaw, 'notes': '; '.join(notes)})
    return rows, counts


def format_report(rows, counts, poses_file, map_info=None):
    lines = [f'poses.yaml : {poses_file}',
             f'peta       : {map_info["path"]} ({map_info["w"]}x{map_info["h"]} px, '
             f'{map_info["res"]} m/px)' if map_info else 'peta       : (tidak ada -> cek okupansi dilewati)',
             '',
             f'{"STATUS":<7} {"NAMA":<14} {"X":>8} {"Y":>8} {"YAW":>7}  CATATAN',
             '-' * 72]
    for r in rows:
        xy = (f'{r["x"]:>8.3f} {r["y"]:>8.3f}' if r['x'] is not None
              else f'{"-":>8} {"-":>8}')
        yaw = f'{r["yaw"]:>6.1f}°' if r['yaw'] is not None else f'{"-":>7}'
        lines.append(f'{r["status"]:<7} {r["name"]:<14} {xy} {yaw}  {r["notes"]}')
    total = len(rows)
    lines += ['-' * 72,
              f'{total} preset: {counts["OK"]} OK, {counts["WARN"]} WARN, {counts["FAIL"]} FAIL']
    return '\n'.join(lines)


def _ament_share(pkg):
    """share dir paket (install), '' bila ament_index tidak tersedia (source langsung)."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory(pkg)
    except Exception:
        return ''


def default_poses_file():
    """Lokasi config/poses.yaml: share dir (hasil build) atau source tree (fallback)."""
    share = _ament_share('spawn')
    if share:
        p = os.path.join(share, 'config', 'poses.yaml')
        if os.path.isfile(p):
            return p
    here = os.path.dirname(os.path.realpath(__file__))  # src/spawn/scripts
    return os.path.normpath(os.path.join(here, '..', 'config', 'poses.yaml'))


def default_map_file():
    """Peta navigation untuk cek okupansi, '' bila tidak ketemu."""
    share = _ament_share('navigation')
    if share:
        p = os.path.join(share, 'maps', 'map.yaml')
        if os.path.isfile(p):
            return p
    here = os.path.dirname(os.path.realpath(__file__))  # src/spawn/scripts
    cand = os.path.normpath(os.path.join(here, '..', '..', 'navigation', 'maps', 'map.yaml'))
    return cand if os.path.isfile(cand) else ''
