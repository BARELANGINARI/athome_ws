#include "luggage_task/task.hpp"

#include <cmath>
#include <sstream>

namespace luggage_task {

// --- util --------------------------------------------------------------------
double wrap180(double angle_deg)
{
  while (angle_deg > 180.0) {angle_deg -= 360.0;}
  while (angle_deg < -180.0) {angle_deg += 360.0;}
  return angle_deg;
}

std::string trim(const std::string & s)
{
  const char * ws = " \t\r\n";
  const auto a = s.find_first_not_of(ws);
  if (a == std::string::npos) {return "";}
  const auto b = s.find_last_not_of(ws);
  return s.substr(a, b - a + 1);
}

std::vector<std::string> split(const std::string & s, char delim)
{
  std::vector<std::string> out;
  std::string cur;
  std::istringstream ss(s);
  while (std::getline(ss, cur, delim)) {
    out.push_back(trim(cur));
  }
  return out;
}

const char * stateName(State s)
{
  switch (s) {
    case State::START: return "START";
    case State::WAIT_WAKEWORD: return "WAIT_WAKEWORD";
    case State::WAIT_FOLLOW: return "WAIT_FOLLOW";
    case State::DETECT_PERSON: return "DETECT_PERSON";
    case State::PERSON_POSE: return "PERSON_POSE";
    case State::DETECT_GESTURE: return "DETECT_GESTURE";
    case State::TARGET_DIR: return "TARGET_DIR";
    case State::SEARCH_BAG: return "SEARCH_BAG";
    case State::BAG_POSE: return "BAG_POSE";
    case State::ROTATE_TO_BAG: return "ROTATE_TO_BAG";
    case State::APPROACH: return "APPROACH";
    case State::WAIT_TURN: return "WAIT_TURN";
    case State::TRACK: return "TRACK";
    case State::FOLLOW: return "FOLLOW";
    case State::RECOVERY: return "RECOVERY";
    case State::RETURN_START: return "RETURN_START";
    case State::DONE: return "DONE";
    case State::ABORT: return "ABORT";
  }
  return "?";
}

void World::gotoState(State s, double now, const std::string & text)
{
  state = s;
  state_t0 = now;
  status_text = text;
  status_dirty = true;
  cmd = MotionCmd{};
  tick.clear();
}

// --- parse person mentah ------------------------------------------------------
namespace {

bool parseDouble(const std::string & s, double & out)
{
  try {
    size_t pos = 0;
    out = std::stod(s, &pos);
    return pos == trim(s).size();
  } catch (...) {
    return false;
  }
}

// Klasifikasi lengan dari keypoint mentah: bahu->pergelangan menunjuk
// diagonal bawah. Idx BODY_34: kiri bahu 5/pergelangan 7, kanan 12/14.
std::string classifyArmRaw(
  const std::vector<double> & kpt, int shoulder, int wrist)
{
  constexpr double kThreshold = 0.5;  // sama dengan default zed
  const auto get = [&](int i, double & x, double & y) {
    const size_t base = static_cast<size_t>(i) * 2;
    if (base + 1 >= kpt.size()) {return false;}
    x = kpt[base];
    y = kpt[base + 1];
    return x >= 0.0 && y >= 0.0;
  };
  double sx = 0.0, sy = 0.0, wx = 0.0, wy = 0.0;
  if (!get(shoulder, sx, sy) || !get(wrist, wx, wy)) {return "";}
  const double dx = wx - sx;
  const double dy = wy - sy;
  if (dy <= 0.0) {return "";}
  if (std::fabs(dx) < kThreshold * dy) {return "";}
  return dx < 0.0 ? "DOWN_LEFT" : "DOWN_RIGHT";
}

// Hitung ulang label gestur dari keypoint saat ini. Dipakai parsePersons
// dan WAJIB dipanggil lagi sehabis transform (mirror) supaya label tidak basi.
void refreshGesture(PersonInfo & info)
{
  info.gesture.clear();
  const std::string left = classifyArmRaw(info.keypoints, 5, 7);
  const std::string right = classifyArmRaw(info.keypoints, 12, 14);
  if (!left.empty()) {info.gesture += "LEFT_HAND_" + left;}
  if (!right.empty()) {
    if (!info.gesture.empty()) {info.gesture += " | ";}
    info.gesture += "RIGHT_HAND_" + right;
  }
  if (info.gesture.empty()) {info.gesture = "NO_DIAGONAL_DOWN";}
}

}  // namespace

std::vector<PersonInfo> parsePersons(const std::string & data)
{
  const std::string s = trim(data);
  if (s.empty() || s == "NO_PERSON") {return {};}

  std::vector<PersonInfo> persons;
  for (const auto & entry : split(s, ';')) {
    if (entry.empty()) {continue;}
    // "person:x,y,w,h|x,y,z|yaw,rel|kpt=x1,y1,...,x34,y34"
    const auto parts = split(entry, '|');
    if (parts.size() < 4) {continue;}
    PersonInfo info;
    try {
      std::string head = parts[0];
      const auto colon = head.find(':');
      if (colon != std::string::npos) {head = trim(head.substr(colon + 1));}
      const auto b = split(head, ',');
      if (b.size() != 4) {continue;}
      info.bbox_x = std::stoi(b[0]);
      info.bbox_y = std::stoi(b[1]);
      info.bbox_w = std::stoi(b[2]);
      info.bbox_h = std::stoi(b[3]);
      info.has_bbox = true;
      const auto p = split(parts[1], ',');
      if (p.size() == 3) {
        double x = 0.0, y = 0.0, z = 0.0;
        if (parseDouble(p[0], x) && parseDouble(p[1], y) && parseDouble(p[2], z)) {
          info.pos_x = x;
          info.pos_y = y;
          info.pos_z = z;
          info.has_pos = std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
        }
      }
      const auto y = split(parts[2], ',');
      if (y.size() == 2) {
        double yaw = 0.0, rel = 0.0;
        if (parseDouble(y[0], yaw)) {info.yaw = yaw;}
        if (parseDouble(y[1], rel)) {info.rel = rel;}
      }
      std::string k = parts[3];
      const auto eq = k.find('=');
      if (eq != std::string::npos) {k = k.substr(eq + 1);}
      for (const auto & tok : split(k, ',')) {
        double v = 0.0;
        if (parseDouble(tok, v)) {info.keypoints.push_back(v);}
      }
    } catch (...) {
      continue;
    }
    if (info.rel.has_value()) {
      const double a = std::fabs(*info.rel);
      if (a <= 30.0) {info.orientation = "FRONT";}
      else if (a >= 150.0) {info.orientation = "BACK";}
      else if (*info.rel > 0.0) {info.orientation = "SIDE_LEFT";}
      else {info.orientation = "SIDE_RIGHT";}
    }
    refreshGesture(info);
    persons.push_back(info);
  }
  return persons;
}

std::string gestureToDir(const std::string & gesture)
{
  if (gesture.find("DOWN_LEFT") != std::string::npos) {return "kiri";}
  if (gesture.find("DOWN_RIGHT") != std::string::npos) {return "kanan";}
  return "bawah";
}

std::vector<BagInfo> parseBags(const std::string & data)
{
  std::vector<BagInfo> out;
  const std::string s = trim(data);
  if (s.empty()) {return out;}
  for (const auto & entry : split(s, ';')) {
    if (entry.empty()) {continue;}
    // "bag:x,y,w,h|x,y,z|conf" (3D bisa "nan,nan,nan").
    const auto parts = split(entry, '|');
    if (parts.size() < 2) {continue;}
    BagInfo b;
    try {
      std::string head = parts[0];
      const auto colon = head.find(':');
      if (colon != std::string::npos) {head = trim(head.substr(colon + 1));}
      const auto v = split(head, ',');
      if (v.size() != 4) {continue;}
      b.bbox_x = std::stoi(v[0]);
      b.bbox_y = std::stoi(v[1]);
      b.bbox_w = std::stoi(v[2]);
      b.bbox_h = std::stoi(v[3]);
      b.has_bbox = true;
      const auto p = split(parts[1], ',');
      if (p.size() == 3) {
        double x = 0.0, y = 0.0, z = 0.0;
        if (parseDouble(p[0], x) && parseDouble(p[1], y) && parseDouble(p[2], z) &&
          std::isfinite(x) && std::isfinite(y) && std::isfinite(z))
        {
          b.pos_x = x;
          b.pos_y = y;
          b.pos_z = z;
          b.has_pos = true;
        }
      }
      if (parts.size() >= 3) {
        double c = 0.0;
        if (parseDouble(parts[2], c)) {b.score = c;}
      }
    } catch (...) {
      continue;
    }
    out.push_back(b);
  }
  return out;
}

// --- taktik -------------------------------------------------------------------
bool isPointing(const PersonInfo & p)
{
  return p.gesture.find("DOWN_LEFT") != std::string::npos ||
    p.gesture.find("DOWN_RIGHT") != std::string::npos;
}

std::string pointingDir(const PersonInfo & p)
{
  return gestureToDir(p.gesture);
}

int pickOperator(const std::vector<PersonInfo> & persons)
{
  if (persons.empty()) {return -1;}
  // Orang PALING DEKAT = yg terdeteksi (zed sudah urut terdekat dulu).
  // Range 3D terkecil menang; bila depth NaN semua, fallback bbox terlebar.
  int best = -1;
  double best_range = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < persons.size(); ++i) {
    const auto & p = persons[i];
    if (p.has_pos && std::isfinite(p.pos_x) && std::isfinite(p.pos_y)) {
      const double r = std::hypot(p.pos_x, p.pos_y);
      if (r < best_range) {
        best_range = r;
        best = static_cast<int>(i);
      }
    }
  }
  if (best >= 0) {return best;}
  int best_w = -1;
  best = 0;
  for (size_t i = 0; i < persons.size(); ++i) {
    if (persons[i].has_bbox && persons[i].bbox_w > best_w) {
      best_w = persons[i].bbox_w;
      best = static_cast<int>(i);
    }
  }
  return best;
}

// Pilih tas sesuai zona arah telunjuk: "kiri" -> bbox kiri tengah gambar,
// "kanan" -> kanan, else terbaik. Return "x,y,w,h" terpilih.
std::optional<std::string> pickBag(
  const std::string & bag_msg, int img_w, const std::string & dir)
{
  const std::string s = trim(bag_msg);
  if (s.empty()) {return std::nullopt;}
  std::optional<std::string> fallback;
  for (const auto & entry : split(s, ';')) {
    if (entry.empty()) {continue;}
    // "bag:x,y,w,h|x,y,z|conf" atau legacy "x,y,w,h[|...]".
    std::string head = entry;
    const auto bar = head.find('|');
    if (bar != std::string::npos) {head = trim(head.substr(0, bar));}
    const auto colon = head.find(':');
    if (colon != std::string::npos) {head = trim(head.substr(colon + 1));}
    const auto v = split(head, ',');
    if (v.size() != 4) {continue;}
    try {
      const double cx = std::stod(v[0]) + std::stod(v[2]) / 2.0;
      if (!fallback.has_value()) {fallback = head;}
      const bool left = cx < img_w / 2.0;
      if ((dir == "kiri" && left) || (dir == "kanan" && !left)) {
        return head;  // skor tertinggi di zona (urut skor dari zed)
      }
    } catch (...) {
      continue;
    }
  }
  return fallback;
}

std::optional<double> bagBearing(
  const std::string & box, int img_w, double hfov_deg)
{
  // "x,y,w,h" -> bearing deg thd tengah gambar (+ = kiri).
  const auto v = split(trim(box), ',');
  if (v.size() != 4) {return std::nullopt;}
  try {
    const double cx = std::stod(v[0]) + std::stod(v[2]) / 2.0;
    return (img_w / 2.0 - cx) / img_w * hfov_deg;
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<double> operatorBearing(
  const std::vector<PersonInfo> & persons, int img_w, double hfov_deg)
{
  // Person BACK (membelakangi = operator jalan di depan) -> bearing.
  for (const auto & p : persons) {
    if (p.orientation != "BACK") {continue;}
    if (!p.has_bbox) {return 0.0;}
    const double cx = p.bbox_x + p.bbox_w / 2.0;
    return (img_w / 2.0 - cx) / img_w * hfov_deg;
  }
  return std::nullopt;
}

std::optional<double> approachBearing(
  const std::vector<PersonInfo> & persons, int img_w, double hfov_deg)
{
  // Person FRONT terdekat tengah; fallback bbox terlebar.
  const PersonInfo * best = nullptr;
  double best_dist = 0.0;
  for (const auto & p : persons) {
    if (p.orientation != "FRONT" || !p.has_bbox) {continue;}
    const double cx = p.bbox_x + p.bbox_w / 2.0;
    const double dist = std::fabs(cx - img_w / 2.0);
    if (best == nullptr || dist < best_dist) {
      best = &p;
      best_dist = dist;
    }
  }
  if (best == nullptr) {
    int best_w = -1;
    for (const auto & p : persons) {
      if (p.has_bbox && p.bbox_w > best_w) {
        best_w = p.bbox_w;
        best = &p;
      }
    }
  }
  if (best == nullptr) {return std::nullopt;}
  if (!best->has_bbox) {return 0.0;}
  const double cx = best->bbox_x + best->bbox_w / 2.0;
  return (img_w / 2.0 - cx) / img_w * hfov_deg;
}

FollowGoal followGoal(
  double robot_x, double robot_y, double robot_yaw,
  double bearing_deg, double front_range_m,
  double follow_stop_m, double min_goal_m)
{
  FollowGoal g;
  // bearing + = kiri, yaw ROS CCW+ -> goal kiri = yaw + bearing.
  const double theta = robot_yaw + bearing_deg * M_PI / 180.0;
  const double len = front_range_m - follow_stop_m;
  if (len < min_goal_m) {return g;}
  g.valid = true;
  g.x = robot_x + std::cos(theta) * len;
  g.y = robot_y + std::sin(theta) * len;
  g.theta = theta;
  return g;
}

// --- fuse ----------------------------------------------------------------------
void integrate(World * world, const Snapshot & snap)
{
  if (world == nullptr) {return;}
  // Person mentah -> lock terlihat.
  if (!snap.persons_raw.empty()) {world->gesture_seen = true;}
  world->persons = parsePersons(snap.persons_raw);
  // Mirror: kanan-asli tampil kanan (x' = img_w - x) agar label kiri/kanan
  // mengikuti perspektif person. Diterapkan di sini supaya logika gestur +
  // tampilan konsisten dalam frame yang sama.
  if (world->params.mirror) {
    const double W = static_cast<double>(world->params.img_w);
    for (auto & person : world->persons) {
      if (person.has_bbox) {
        person.bbox_x = static_cast<int>(W - person.bbox_x - person.bbox_w);
      }
      for (size_t k = 0; k + 1 < person.keypoints.size(); k += 2) {
        double & kx = person.keypoints[k];
        if (kx >= 0.0 && std::isfinite(kx)) {kx = W - kx;}
      }
      refreshGesture(person);  // label wajib dihitung ulang setelah mirror
    }
  }
  // Tas YOLO (semua + distance), mirror ikut persons bila aktif.
  world->bags = parseBags(snap.bag);
  if (world->params.mirror) {
    const double W = static_cast<double>(world->params.img_w);
    for (auto & bag : world->bags) {
      if (bag.has_bbox) {
        bag.bbox_x = static_cast<int>(W - bag.bbox_x - bag.bbox_w);
      }
    }
  }
  // Tas: latch deteksi; simulasi bila diaktifkan (tanpa YOLO).
  if (!snap.bag.empty()) {
    const auto pick = pickBag(snap.bag, world->params.img_w, world->target_dir);
    if (pick.has_value()) {world->bag_box = *pick;}
  } else if (world->bag_box.empty() && world->params.simulate_bag &&
    world->state == State::SEARCH_BAG &&
    world->elapsed(snap.now) > world->params.simulate_sec)
  {
    int cx = world->params.img_w / 2;
    if (world->target_dir == "kiri") {
      cx = static_cast<int>(world->params.img_w * 0.25);
    } else if (world->target_dir == "kanan") {
      cx = static_cast<int>(world->params.img_w * 0.75);
    }
    std::ostringstream ss;
    ss << (cx - 60) << ",300,120,160";
    world->bag_box = ss.str();
  }
  // Operator = person terpilih + range 3D (fallback lidar di decide).
  world->op_visible = !world->persons.empty();
  world->op_range_m = -1.0;
  world->op_bearing_deg = 0.0;
  if (!world->persons.empty()) {
    const int pick = pickOperator(world->persons);
    if (pick >= 0) {
      const auto & p = world->persons[static_cast<size_t>(pick)];
      if (p.has_pos && std::isfinite(p.pos_x) && std::isfinite(p.pos_y)) {
        world->op_range_m = std::hypot(p.pos_x, p.pos_y);
        world->op_bearing_deg = std::atan2(-p.pos_y, p.pos_x) * 180.0 / M_PI;
      }
    }
  }
  const auto bearing = operatorBearing(
    world->persons, world->params.img_w, world->params.hfov);
  world->follow_bearing_valid = bearing.has_value();
  world->follow_bearing_deg = bearing.value_or(0.0);
  // Robot.
  world->yaw = snap.yaw;
  world->robot.have_pose = snap.have_pose;
  world->robot.x = snap.pose_x;
  world->robot.y = snap.pose_y;
  world->robot.yaw = snap.pose_yaw;
  world->robot.have_scan = snap.have_scan;
  world->robot.front_min_range_m = snap.front_min_range;
  if (snap.goal_reached) {world->goal_reached = true;}
  if (!snap.hri_event.empty()) {world->hri_event = snap.hri_event;}
}

// --- state machine ---------------------------------------------------------------
namespace {

void abortTo(World * world, double now, const std::string & reason)
{
  world->gotoState(State::ABORT, now, "ABORT: " + reason);
}

bool needYaw(World * world, double now)
{
  if (!world->yaw) {
    if (world->elapsed(now) > 10.0) {
      abortTo(world, now, "tidak ada /zed/yaw_deg (cek node zed)");
    }
    return false;
  }
  return true;
}

double personRange(const World * world)
{
  double range = std::numeric_limits<double>::infinity();
  if (world->op_visible && world->op_range_m > 0.01) {
    range = world->op_range_m;
  }
  const double lidar = world->robot.front_min_range_m;
  if (std::isfinite(lidar)) {range = std::min(range, lidar);}
  return range;
}

}  // namespace

void decide(World * world, double now, bool nav_ready)
{
  if (world == nullptr) {return;}
  world->tick.clear();
  switch (world->state) {
    case State::START: {
      // B: inisialisasi — tunggu kamera (topik mentah) + lidar.
      if (world->gesture_seen && world->robot.have_scan) {
        if (world->params.use_hri) {
          world->gotoState(State::WAIT_WAKEWORD, now, "Kamera & lidar siap. Menunggu wakeword 'hello robot'...");
        } else {
          world->gotoState(State::DETECT_PERSON, now, "Kamera + lidar siap. Deteksi person...");
        }
      } else if (world->elapsed(now) > 15.0) {
        std::string missing;
        if (!world->gesture_seen) {missing += "/zed/person_keypoints (node zed) ";}
        if (!world->robot.have_scan) {missing += "/scan (lidar)";}
        abortTo(world, now, "input tidak datang: " + missing);
      }
      break;
    }
    case State::WAIT_WAKEWORD: {
      if (world->hri_event == "WAKEWORD_DETECTED") {
        world->hri_event.clear();
        world->say("Hello, I'm Barelang Inari, I want to help you to carry the luggage. If you are ready, please guide me to the place, and say follow me.");
        world->gotoState(State::WAIT_FOLLOW, now, "Wakeword terdeteksi. Menunggu perintah 'follow me'...");
      }
      break;
    }
    case State::WAIT_FOLLOW: {
      if (world->hri_event == "FOLLOW_COMMAND") {
        world->hri_event.clear();
        world->say("Following you now.");
        world->last_person_seen = now;
        world->retries_left = world->params.reacquire_retries;
        world->gotoState(State::FOLLOW, now, "Perintah follow diterima. Memulai following...");
      }
      break;
    }
    case State::DETECT_PERSON: {
      // C: deteksi person dari keypoint.
      if (!world->persons.empty()) {
        const int pick = pickOperator(world->persons);
        world->target_person = world->persons[static_cast<size_t>(pick < 0 ? 0 : pick)];
        world->have_target = true;
        world->last_person_seen = now;
        world->gotoState(State::PERSON_POSE, now, "Person terdeteksi. Estimasi pose 3D...");
      } else if (world->elapsed(now) > world->params.person_timeout) {
        abortTo(world, now, "timeout: tidak ada person terdeteksi");
      }
      break;
    }
    case State::PERSON_POSE: {
      // D+E: bbox + depth 3D person. Kunci ulang tiap tick.
      if (world->persons.empty()) {
        world->gotoState(State::DETECT_PERSON, now, "Person hilang, ulangi deteksi.");
        break;
      }
      const int pick = pickOperator(world->persons);
      world->target_person = world->persons[static_cast<size_t>(pick < 0 ? 0 : pick)];
      world->have_target = true;
      world->last_person_seen = now;
      if (!world->target_person.has_pos &&
        world->elapsed(now) > world->params.person_timeout)
      {
        abortTo(world, now, "timeout: depth person tidak tersedia");
        break;
      }
      if (world->target_person.has_pos) {
        std::ostringstream ss;
        ss << "Pose person ok (" << world->target_person.pos_x << " m). Deteksi gestur...";
        world->gotoState(State::DETECT_GESTURE, now, ss.str());
      }
      break;
    }
    case State::DETECT_GESTURE: {
      // F: gestur tangan + arah telunjuk.
      if (!world->persons.empty()) {world->last_person_seen = now;}
      for (const auto & person : world->persons) {
        if (isPointing(person)) {
          world->target_person = person;
          world->have_target = true;
          world->gotoState(State::TARGET_DIR, now, "Tangan menunjuk terdeteksi.");
          break;
        }
      }
      if (world->state == State::DETECT_GESTURE &&
        world->elapsed(now) > world->params.hand_timeout)
      {
        abortTo(world, now, "timeout: tidak ada isyarat tangan menunjuk");
      }
      break;
    }
    case State::TARGET_DIR: {
      // G: arah penunjukan -> zona tas kiri/kanan.
      world->target_dir = world->have_target ?
        pointingDir(world->target_person) : "bawah";
      world->target_dir_dirty = true;
      if (world->params.use_hri) {
        std::string dir_en = (world->target_dir == "kiri") ? "left" : "right";
        world->say("I see you are pointing to the " + dir_en + " side. Moving to target bag.");
      }
      if (world->params.gesture_only) {
        // Mode gesture-dulu: tampilkan arah lalu balik deteksi lagi.
        std::ostringstream ss;
        ss << "Arah target: " << world->target_dir << " (gesture-only, ulangi deteksi)";
        world->gotoState(State::DETECT_GESTURE, now, ss.str());
        break;
      }
      std::ostringstream ss;
      ss << "Arah target: " << world->target_dir << ". Cari tas...";
      world->gotoState(State::SEARCH_BAG, now, ss.str());
      break;
    }
    case State::SEARCH_BAG: {
      // H/I/J: tas di zona arah (dipilih saat integrate).
      if (!world->bag_box.empty()) {
        world->gotoState(State::BAG_POSE, now, "Tas terdeteksi: " + world->bag_box);
      } else if (world->elapsed(now) > world->params.search_timeout) {
        abortTo(world, now, "timeout: tas tidak ditemukan");
      }
      break;
    }
    case State::BAG_POSE: {
      // K/L/M: nama+bbox+3D tas + yaw -> bearing putar.
      const auto bearing = bagBearing(
        world->bag_box, world->params.img_w, world->params.hfov);
      if (!bearing) {
        abortTo(world, now, "format deteksi tas salah: " + world->bag_box);
        break;
      }
      world->bag_bearing = *bearing;
      if (std::fabs(world->bag_bearing) < world->params.rotate_tol) {
        world->gotoState(
          State::APPROACH, now,
          "Tas sudah di depan (bearing " + std::to_string(world->bag_bearing) + " deg).");
      } else if (!needYaw(world, now)) {
        break;
      } else {
        world->rotate_start_yaw = *world->yaw;
        world->rotate_target = world->bag_bearing;
        std::ostringstream ss;
        ss << "Estimasi bearing tas " << world->bag_bearing << " deg.";
        world->gotoState(State::ROTATE_TO_BAG, now, ss.str());
      }
      break;
    }
    case State::ROTATE_TO_BAG: {
      // N/O: putar menghadap tas sampai toleransi.
      if (!needYaw(world, now)) {break;}
      const double turned = wrap180(*world->yaw - world->rotate_start_yaw);
      const double remaining = world->rotate_target - turned;
      if (std::fabs(remaining) < world->params.rotate_tol) {
        world->gotoState(State::APPROACH, now, "Menghadap tas. Maju ke operator...");
      } else if (world->elapsed(now) > 30.0) {
        abortTo(world, now, "timeout rotasi ke tas");
      } else {
        const double direction = remaining > 0 ? 1.0 : -1.0;
        world->cmd.vx = 0.0;
        world->cmd.w = direction * world->params.rotate_speed;
        world->cmd.stop = false;
      }
      break;
    }
    case State::APPROACH: {
      // P/Q/R: maju perlahan ke operator/tas sampai <= 0.5 m, lalu berhenti.
      const double range = personRange(world);
      if (range <= world->params.approach_stop_m) {
        world->cmd = MotionCmd{};
        if (world->params.use_hri) {
          world->say("Thank you, I have the bag now, and I will back to starting point.");
          world->gotoState(State::RETURN_START, now, "Tas tercapai. Kembali ke titik awal (0.0, 0.0)...");
          break;
        }
        std::ostringstream ss;
        ss << "Sudah 0.5 m dari operator (" << range << " m). Tunggu membelakangi...";
        world->retries_left = world->params.reacquire_retries;
        world->gotoState(State::WAIT_TURN, now, ss.str());
        break;
      }
      if (!world->robot.have_scan && world->op_range_m <= 0.01) {
        if (world->elapsed(now) > 10.0) {
          abortTo(world, now, "tidak ada depth person maupun /scan");
        }
        break;
      }
      if (world->elapsed(now) > world->params.approach_timeout) {
        abortTo(world, now, "timeout: operator masih >0.5m");
        break;
      }
      const auto bearing = approachBearing(
        world->persons, world->params.img_w, world->params.hfov);
      if (!bearing) {break;}  // tunggu person (tetap diam)
      double w = *bearing * M_PI / 180.0 * 1.5;
      if (w > world->params.rotate_speed) {w = world->params.rotate_speed;}
      if (w < -world->params.rotate_speed) {w = -world->params.rotate_speed;}
      world->cmd.vx = world->params.approach_speed;
      world->cmd.w = w;
      world->cmd.stop = false;
      if (std::isfinite(range)) {
        std::ostringstream ss;
        ss << "Mendekati operator... " << range << " m (target 0.5m)";
        world->tick = ss.str();
      } else {
        world->tick = "Mendekati operator... menunggu depth";
      }
      break;
    }
    case State::WAIT_TURN: {
      // T: operator membelakangi robot (BACK) sebelum tracking.
      bool back = false;
      for (const auto & p : world->persons) {
        if (p.orientation == "BACK") {
          world->target_person = p;
          world->have_target = true;
          back = true;
          break;
        }
      }
      if (back) {
        world->gotoState(State::TRACK, now, "Operator membelakangi. Tracking...");
      } else if (world->elapsed(now) > world->params.person_timeout) {
        --world->retries_left;
        if (world->retries_left > 0) {
          world->gotoState(State::DETECT_PERSON, now, "Belum membelakangi, ulangi kunci person.");
        } else {
          abortTo(world, now, "timeout: operator tak kunjung membelakangi");
        }
      }
      break;
    }
    case State::TRACK: {
      // U/V: keypoint belakang + tracking identitas/posisi.
      if (!world->persons.empty()) {
        world->last_person_seen = now;
        if (world->follow_bearing_valid) {
          world->follow_t0 = now;
          world->gotoState(State::FOLLOW, now, "Mengikuti operator dari belakang.");
        } else {
          world->tick = "Menunggu punggung operator...";
        }
      } else if (world->elapsed(now) > world->params.person_timeout) {
        --world->retries_left;
        if (world->retries_left > 0) {
          world->gotoState(State::DETECT_PERSON, now, "Operator hilang, kunci ulang.");
        } else {
          abortTo(world, now, "timeout: operator tidak ditemukan kembali");
        }
      }
      break;
    }
    case State::FOLLOW: {
      // W/X/Y/Z: ikut + jaga 0.5 m; selesai bila perintah suara arrived / goal tercapai / durasi habis.
      if (world->hri_event == "ARRIVED_COMMAND" || world->goal_reached) {
        world->hri_event.clear();
        world->cmd = MotionCmd{};
        world->say("Please point to your luggage, and I will drive towards it.");
        world->gotoState(State::DETECT_GESTURE, now, "Tiba di tujuan. Menunggu isyarat tunjuk tas...");
        break;
      }
      if (world->params.follow_duration > 0.0 && world->elapsed(now) > world->params.follow_duration) {
        if (world->params.use_hri) {
          world->cmd = MotionCmd{};
          world->say("Please point to your luggage, and I will drive towards it.");
          world->gotoState(State::DETECT_GESTURE, now, "Durasi follow habis. Menunggu isyarat tunjuk tas...");
        } else {
          world->gotoState(State::DONE, now, "Durasi follow habis. Task selesai.");
        }
        break;
      }
      if (world->persons.empty()) {
        if (now - world->last_person_seen > world->params.lost_timeout) {
          if (world->params.use_hri) {
            world->recovery_attempts = 1;
            world->last_recovery_speak = now;
            world->say("Please stop, and stand in my front camera, because I lost you.");
            world->gotoState(State::RECOVERY, now, "Operator hilang. Meminta berhenti...");
          } else {
            world->gotoState(State::TRACK, now, "Operator hilang, lacak ulang...");
          }
        }
        break;
      }
      world->last_person_seen = now;

      // Sinkron person_follower.py: jika operator menjauh melebihi max_follow_range (default 3.0 m)
      double dist_to_person = -1.0;
      for (const auto & p : world->persons) {
        if (p.has_pos) {
          dist_to_person = std::hypot(p.pos_x, p.pos_y);
          break;
        }
      }
      if (dist_to_person > world->params.max_follow_range) {
        world->cmd = MotionCmd{};
        if (world->params.use_hri) {
          world->recovery_attempts = 1;
          world->last_recovery_speak = now;
          world->say("Please stop, you are too far away, I lost you.");
          world->gotoState(
            State::RECOVERY, now,
            "Operator terlalu jauh (" + std::to_string(dist_to_person) + " m > " +
            std::to_string(world->params.max_follow_range) + " m). Berhenti...");
        } else {
          abortTo(
            world, now,
            "operator terlalu jauh (" + std::to_string(dist_to_person) + " m > " +
            std::to_string(world->params.max_follow_range) + " m)");
        }
        break;
      }

      if (!world->robot.have_pose) {
        if (world->elapsed(now) > 15.0) {
          abortTo(world, now, "TF peta hilang (cek navigation)");
        }
        break;
      }
      if (!nav_ready && world->elapsed(now) > world->params.nav_ready_timeout) {
        abortTo(world, now, "Nav2 hilang (cek navigation)");
      }
      break;
    }
    case State::RECOVERY: {
      world->cmd = MotionCmd{};
      bool person_in_range = false;
      if (!world->persons.empty()) {
        for (const auto & p : world->persons) {
          if (p.has_pos) {
            double d = std::hypot(p.pos_x, p.pos_y);
            if (d <= world->params.max_follow_range) {
              person_in_range = true;
              break;
            }
          } else {
            person_in_range = true;
            break;
          }
        }
      }
      if (person_in_range) {
        world->last_person_seen = now;
        world->say("I see you. Resuming following.");
        world->gotoState(State::FOLLOW, now, "Operator terlihat kembali dalam jangkauan. Melanjutkan follow...");
        break;
      }
      if (now - world->last_recovery_speak >= 3.0) {
        world->last_recovery_speak = now;
        world->recovery_attempts++;
        if (world->recovery_attempts <= 5) {
          world->say("Please stop, and stand in my front camera, because I lost you.");
        } else if (world->recovery_attempts > 8) {
          abortTo(world, now, "Operator tidak ditemukan setelah panggilan recovery.");
        }
      }
      break;
    }
    case State::RETURN_START: {
      if (world->goal_reached || world->elapsed(now) > 60.0) {
        world->say("I've already finished the task.");
        world->gotoState(State::DONE, now, "Tiba di titik awal. Task selesai.");
      }
      break;
    }
    case State::DONE:
    case State::ABORT:
      break;
  }
}

}  // namespace luggage_task
