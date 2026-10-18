#!/usr/bin/env bash
# Diagnosa cepat rantai navigasi: TF -> scan -> odom -> AMCL -> costmap.
#
# Pakai:
#   ros2 run navigation check_system
#   bash src/navigation/scripts/check_system.sh
#
# Exit 0 = semua sehat. Exit 1 = ada masalah (lihat ringkasan akhir).
set -uo pipefail

OK=0
BAD=0

line() { printf '%s\n' "------------------------------------------------"; }
ok()   { printf '  [OK]    %s\n' "$1"; OK=$((OK+1)); }
bad()  { printf '  [GAGAL] %s\n' "$1"; BAD=$((BAD+1)); }
warn() { printf '  [WARN]  %s\n' "$1"; }

hz() {  # hz <topic> -> "N.NN" Hz atau "tidak ada"
  local t=$1 out
  out=$(timeout 5 ros2 topic hz "$t" 2>/dev/null | grep -oP 'average rate: \K[0-9.]+')
  if [ -z "$out" ]; then echo "tidak ada"; else echo "$out"; fi
}

has_node() { ros2 node list 2>/dev/null | grep -qx -e "$1" -e "/$1"; }

echo "=== 1. NODE WAJIB ==="
line
for n in nucleo amcl map_server planner_server controller_server bt_navigator; do
  if has_node "$n"; then ok "node $n hidup"; else bad "node $n TIDAK ada"; fi
done

echo
echo "=== 2. TOPIK SENSOR & ODOM ==="
line
s=$(hz /scan)
case "$s" in
  "tidak ada") bad "/scan tidak terbit (lidar tidak jalan / salah port)" ;;
  0.*)        bad "/scan terbit tapi ~${s} Hz - sangat lambat, cek scan_mode" ;;
  *)          ok "/scan ${s} Hz" ;;
esac
o=$(hz /odom)
[ "$o" = "tidak ada" ] && bad "/odom tidak terbit (nucleo?)" || ok "/odom ${o} Hz"

echo
echo "=== 3. RANTAI TF (ini kunci utama) ==="
line
for pair in "map odom" "odom base_footprint" "base_footprint base_link" "base_link laser_frame"; do
  set -- $pair
  if timeout 6 ros2 run tf2_ros tf2_echo "$1" "$2" 2>&1 | grep -q "Translation:"; then
    ok "TF $1 -> $2 ADA"
  else
    bad "TF $1 -> $2 HILANG  <-- ini penyebab utama display beku"
  fi
done

echo
echo "=== 4. LOKALISASI AMCL ==="
line
if has_node amcl; then
  st=$(timeout 5 ros2 lifecycle get /amcl 2>/dev/null)
  case "$st" in
    *active*) ok "amcl $st" ;;
    *)        bad "amcl belum aktif ($st)" ;;
  esac
  if timeout 4 ros2 topic echo /amcl_pose --once 2>/dev/null | grep -q position; then
    ok "AMCLEstimasi pose terkirim (terlocalisasi)"
  else
    bad "AMCL belum localized - robot belum di-set pose awal"
  fi
else
  bad "amcl tidak ada"
fi

echo
echo "=== 4b. KESEHATAN LOKALISASI (scan nempel peta?) ==="
line
# Confidence AMCL = covariance x. Makin kecil = makin yakin.
# Kalau besar, lokalisasi melenceng (scan tidak akan nempel ke peta).
conf=$(timeout 6 ros2 topic echo /amcl_pose --once 2>/dev/null \
  | grep -A 2 'covariance' | grep -oP '^\s*-\s*\K[0-9.]+' | head -1)
if [ -n "$conf" ]; then
  case "$conf" in
    0.000*|0.00[0-3]*) bad "confidence AMCL rendah ($conf) -> lokalisasi tidak yakin / melenceng" ;;
    *)                 ok  "confidence AMCL sehat (cov-x = $conf)" ;;
  esac
else
  warn "tidak bisa baca confidence (amcl_pose belum ada)"
fi
# Jarak odom vs AMCL: kalau jauh = odom drifting jauh dari peta.
num() {  # num -> angka atau string kosong (buang baris WARNING ros2)
  grep -oP '^\s*-?\K[0-9.]+(?=\s*$)' | head -1
}
o_x=$(timeout 5 ros2 topic echo /odom --once --field pose.pose.position.x 2>/dev/null | num)
a_x=$(timeout 5 ros2 topic echo /amcl_pose --once --field pose.pose.position.x 2>/dev/null | num)
if [ -n "$o_x" ] && [ -n "$a_x" ]; then
  echo "         odom.x = $o_x m | amcl.x = $a_x m"
else
  echo "         (odom/amcl belum terbaca - robot mungkin diam)"
fi

echo
echo "=== 5. COSTMAP ==="
line
for t in /global_costmap/costmap /local_costmap/costmap; do
  if timeout 5 ros2 topic echo "$t" --once --field info.width 2>/dev/null | grep -qE '^[0-9]'; then
    ok "$t terbit"
  else
    bad "$t tidak terbit (costmap belum aktif / no map received)"
  fi
done

echo
echo "=== 6. SOURCE /goto ==="
line
if has_node place_navigator; then
  ok "place_navigator hidup"
else
  warn "place_navigator tidak jalan (kalau tidak pakai /goto, ini normal)"
fi

echo
echo "=== 7. BENTROKAN cmd_vel (duplikat publisher) ==="
line
n=$(timeout 5 ros2 topic info /cmd_vel 2>/dev/null | grep -oP 'Publisher count: \K[0-9]+')
if [ -z "$n" ]; then
  warn "tidak ada publisher /cmd_vel (normal kalau robot belum hidup)"
elif [ "$n" -gt 1 ]; then
  bad "/cmd_vel punya $n publisher (további + collision_monitor Nav2)
         -> robot dapat perintah bercampur, planning kacau, sering 'error
            untuk ke target'.
         FIX: teleop harus ke channel override 'joy_vel' (bukan cmd_vel),
         atau matikan Nav2 sementara telop."
else
  ok "/cmd_vel publisher tunggal ($n)"
fi

echo
line
printf 'RINGKASAN: %d OK, %d GAGAL\n' "$OK" "$BAD"
[ "$BAD" -eq 0 ] && echo "Semua sehat." || echo "Ada masalah - perbaiki bagian GAGAL di atas."
exit $([ "$BAD" -eq 0 ] && echo 0 || echo 1)