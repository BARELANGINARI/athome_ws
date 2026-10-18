#!/usr/bin/env python3
"""Validasi poses.yaml tanpa ROS: tabel status semua preset + cek di peta.

  ros2 run spawn check_poses                      # pakai config/poses.yaml + peta navigation
  ros2 run spawn check_poses -- /path/poses.yaml  # file lain
  python3 src/spawn/scripts/check_poses.py        # langsung, tanpa build

Exit code 0 = semua preset OK/WARN, 1 = ada FAIL (atau YAML tidak terbaca).
Untuk uji publish ke /initialpose (lihat robot pindah di RViz):
  ros2 launch spawn test_poses.py
"""
import os
import sys

import pose_check


def main() -> int:
    if len(sys.argv) > 1:
        poses_file = os.path.abspath(sys.argv[1])
    else:
        poses_file = pose_check.default_poses_file()

    presets, err = pose_check.load_presets(poses_file)
    if err:
        print(f'[FAIL] {err}')
        return 1
    map_info = pose_check.load_map(pose_check.default_map_file())
    rows, counts = pose_check.evaluate(presets, map_info)
    print(pose_check.format_report(rows, counts, poses_file, map_info))
    return 1 if counts['FAIL'] else 0


if __name__ == '__main__':
    sys.exit(main())
