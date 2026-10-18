#!/usr/bin/env python3
"""Cek config/places.yaml: nama valid + koordinat free di peta. Tanpa ROS.

  python3 src/navigation/scripts/check_places.py
  ros2 run navigation check_places [path/places.yaml]

Exit code 0 = semua OK, 1 = ada FAIL (nama invalid atau di luar peta).
Mencek:
  1. key sebagai nama ROS service (huruf/angka/underscore, tidak diawali angka)
  2. koordinat berada di area bebas (tidak menabrak dinding, radius 0.15 m)
"""
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from place_navigator import VALID_NAME, default_places_file, load_places  # noqa: E402

CHECK_RADIUS_M = 0.15


def _read_pgm(path):
    """Parser PGM (P5 biner / P2 ascii) minim tanpa dependensi."""
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
    i += 1
    body = data[i:]
    if magic == b'P5':
        return w, h, bytes(body[:w * h])
    if magic == b'P2':
        return w, h, bytes(int(v) for v in body.split()[:w * h])
    raise ValueError(f'format PGM tidak didukung ({magic!r}) di {path}')


def load_map(map_yaml_path):
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


def default_map_file():
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('navigation')
        cand = os.path.join(share, 'maps', 'map.yaml')
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass
    pkg_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    cand = os.path.join(pkg_dir, 'maps', 'map.yaml')
    return cand if os.path.isfile(cand) else ''


def _cell_state(mi, x, y):
    res, h = mi['res'], mi['h']
    mx = int((x - mi['ox']) / res)
    my = int(h - (y - mi['oy']) / res)
    if not (0 <= mx < mi['w'] and 0 <= my < h):
        return 'out'
    v = mi['body'][my * mi['w'] + mx]
    p = (255.0 - v) / 255.0
    if p > mi['occupied_thresh']:
        return 'occupied'
    if p < mi['free_thresh']:
        return 'free'
    return 'unknown'


def check_entry(name, entry, mi):
    notes = []
    status = 'OK'

    if not VALID_NAME.match(name):
        return 'FAIL', None, None, 'nama tidak valid sebagai ROS service ' \
            '(hanya huruf/angka/underscore, tidak boleh diawali angka)'

    if not isinstance(entry, dict):
        return 'FAIL', None, None, 'bukan mapping {x, y, yaw_deg}'

    try:
        x = float(entry.get('x'))
        y = float(entry.get('y'))
    except (TypeError, ValueError):
        return 'FAIL', None, None, f"x/y bukan angka: {entry.get('x')!r}, {entry.get('y')!r}"

    try:
        yaw = float(entry.get('yaw_deg', 0.0))
    except (TypeError, ValueError):
        yaw = 0.0
        notes.append('yaw_deg bukan angka -> dipakai 0')
        status = 'WARN'

    if mi:
        res = mi['res']
        r = int(CHECK_RADIUS_M / res)
        cells = [_cell_state(mi, x + dx * res, y + dy * res)
                 for dx in range(-r, r + 1) for dy in range(-r, r + 1)]
        if 'out' in cells:
            return 'FAIL', x, y, f'di luar area peta'
        n_occ = cells.count('occupied')
        n_unk = cells.count('unknown')
        if n_occ:
            notes.append(f'mentok dinding ({n_occ} sel sekitar terisi)')
            status = 'WARN' if status == 'OK' else status
        elif n_unk:
            notes.append('area belum dipetakan (abu-abu)')
            status = 'WARN' if status == 'OK' else status
        else:
            notes.append('area bebas')

    return status, x, y, '; '.join(notes)


def main() -> int:
    if len(sys.argv) > 1:
        places_file = os.path.abspath(sys.argv[1])
    else:
        places_file = default_places_file()

    places, err = load_places(places_file)
    if err:
        print(f'[FAIL] {err}')
        return 1

    mi = load_map(default_map_file())

    print(f'places.yaml : {places_file}')
    print(f'peta        : {mi["path"] if mi else "(tidak ada -> cek okupansi dilewati)"}')
    print('')
    print(f'{"STATUS":<7} {"NAMA":<15} {"X":>7} {"Y":>7} {"YAW":>7}  CATATAN')
    print('-' * 74)

    counts = {'OK': 0, 'WARN': 0, 'FAIL': 0}
    for name, entry in places.items():
        status, x, y, note = check_entry(str(name), entry, mi)
        counts[status] += 1
        xs = f'{x:>7.2f}' if x is not None else f'{"-":>7}'
        ys = f'{y:>7.2f}' if y is not None else f'{"-":>7}'
        print(f'{status:<7} {str(name):<15} {xs} {ys} {"":>7}  {note}')

    total = len(places)
    print('-' * 74)
    print(f'{total} lokasi: {counts["OK"]} OK, {counts["WARN"]} WARN, {counts["FAIL"]} FAIL')
    return 1 if counts['FAIL'] else 0


if __name__ == '__main__':
    sys.exit(main())