#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>
#include <dynamixel_sdk.h>
#include "calibration_data.h"

// ---------------- REGISTER AX-18A (PROTOCOL 1.0) ----------------
#define ADDR_MODEL_NUMBER       0
#define ADDR_FIRMWARE           2
#define ADDR_CW_ANGLE_LIMIT     6
#define ADDR_CCW_ANGLE_LIMIT    8
#define ADDR_TORQUE_ENABLE      24
#define ADDR_LED                25
#define ADDR_CW_SLOPE           28
#define ADDR_CCW_SLOPE          29
#define ADDR_GOAL_POSITION      30
#define ADDR_MOVING_SPEED       32
#define ADDR_TORQUE_LIMIT       34
#define ADDR_PRESENT_POS        36
#define ADDR_PRESENT_SPEED      38
#define ADDR_PRESENT_LOAD       40
#define ADDR_PRESENT_VOLT       42
#define ADDR_PRESENT_TEMP       43
#define ADDR_MOVING             46

#define PROTOCOL_VERSION        1.0
#define BAUDRATE                1000000
#define DEVICENAME              "/dev/ttyUSB2"

#define GRIPPER_ID              6
#define MAX_ANGLE               300.0
#define MAX_TICKS               1023
#define MAX_OPEN_DEG            80.0    // Sudut buka penuh (~780 ticks, mencakup bukaan fisik 753 ticks)
#define DEFAULT_RELEASE_DEG     65.0    // Sudut buka default

// Parameter Default Adaptive Grasping
struct GraspConfig {
    int    load_threshold   = 180;   // Ambang beban deteksi objek (0-1023, 180 ≈ 17.6%)
    int    step_ticks       = 4;     // Langkah penutupan per iterasi (~1.17 deg)
    int    step_ms          = 40;    // Interval sampling (40 ms = 25 Hz)
    int    stall_ticks      = 10;    // Selisih (Pos - Goal) indikasi tertahan fisik objek
    int    squeeze_ticks    = 5;     // Tambahan penekanan (+ticks ke arah menutup) setelah kontak
    int    confirm_cycles   = 3;     // Siklus berturut-turut untuk validasi kontak
    int    closing_speed    = 60;    // Kecepatan menutup (1-1023, 60 ≈ 6.8 RPM)
    uint8_t compliance_slope = 64;   // Kelenturan pegas PID
};

struct GripperTelemetry {
    uint16_t goal_pos     = 0;
    uint16_t present_pos  = 0;
    uint16_t torque_limit = 0;
    int      signed_load  = 0;   // Negatif = CW (tutup), Positif = CCW (buka)
    int      load_mag     = 0;   // 0 - 1023
    int      speed_signed = 0;
    double   voltage      = 0.0;
    uint8_t  temperature  = 0;
    uint8_t  moving       = 0;
    uint8_t  hw_error     = 0;
    bool     comm_ok      = false;
};

dynamixel::PortHandler   *portHandler   = nullptr;
dynamixel::PacketHandler *packetHandler = nullptr;
GraspConfig g_cfg;
bool g_is_holding = false;
uint16_t g_hold_tick = 510;

// ---------------- KALIBRASI & KONVERSI ----------------
const CalibratedServoConfig *getGripperCalib() {
    for (const auto &c : ARM_CALIBRATION) {
        if (c.id == GRIPPER_ID) return &c;
    }
    return nullptr;
}

int getCenterTick() {
    const auto *c = getGripperCalib();
    return c ? c->center_tick : 510;
}

int getCwLimit() {
    const auto *c = getGripperCalib();
    return c ? c->cw_limit : 490;
}

int getCcwLimit() {
    const auto *c = getGripperCalib();
    return c ? c->ccw_limit : 800;
}

double ticksToDeg(int ticks) {
    return ticks * (MAX_ANGLE / MAX_TICKS);
}

int degToTicks(double deg) {
    return static_cast<int>(std::round(deg / MAX_ANGLE * MAX_TICKS));
}

// 0.0 deg = tertutup rapat (510), +55.0 deg = terbuka penuh (~700)
double rawToRelAngle(uint16_t raw_pos) {
    return ticksToDeg(static_cast<int>(raw_pos) - getCenterTick());
}

uint16_t relAngleToTick(double rel_deg) {
    rel_deg = std::clamp(rel_deg, 0.0, MAX_OPEN_DEG);
    int goal = getCenterTick() + degToTicks(rel_deg);
    return static_cast<uint16_t>(std::clamp(goal, getCwLimit(), getCcwLimit()));
}

// ---------------- PEMBACAAN TELEMETRI & DEBUG ----------------
std::string decodeHwError(uint8_t err) {
    if (err == 0) return "OK";
    std::string s;
    if (err & (1 << 0)) s += "[InputVoltage] ";
    if (err & (1 << 1)) s += "[AngleLimit] ";
    if (err & (1 << 2)) s += "[Overheating] ";
    if (err & (1 << 3)) s += "[Range] ";
    if (err & (1 << 4)) s += "[Checksum] ";
    if (err & (1 << 5)) s += "[Overload!] ";
    if (err & (1 << 6)) s += "[Instruction] ";
    return s;
}

GripperTelemetry readTelemetry() {
    GripperTelemetry t{};
    uint8_t err = 0;
    uint16_t raw_load = 0, raw_speed = 0;
    uint8_t raw_volt = 0;

    if (packetHandler->read2ByteTxRx(portHandler, GRIPPER_ID, ADDR_PRESENT_POS, &t.present_pos, &err) != COMM_SUCCESS) {
        t.comm_ok = false;
        return t;
    }
    t.hw_error |= err;
    t.comm_ok = true;

    packetHandler->read2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, &t.goal_pos, &err);
    t.hw_error |= err;
    packetHandler->read2ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_LIMIT, &t.torque_limit, &err);
    t.hw_error |= err;
    packetHandler->read2ByteTxRx(portHandler, GRIPPER_ID, ADDR_PRESENT_SPEED, &raw_speed, &err);
    t.hw_error |= err;
    packetHandler->read2ByteTxRx(portHandler, GRIPPER_ID, ADDR_PRESENT_LOAD, &raw_load, &err);
    t.hw_error |= err;
    packetHandler->read1ByteTxRx(portHandler, GRIPPER_ID, ADDR_PRESENT_VOLT, &raw_volt, &err);
    t.hw_error |= err;
    packetHandler->read1ByteTxRx(portHandler, GRIPPER_ID, ADDR_PRESENT_TEMP, &t.temperature, &err);
    t.hw_error |= err;
    packetHandler->read1ByteTxRx(portHandler, GRIPPER_ID, ADDR_MOVING, &t.moving, &err);
    t.hw_error |= err;

    t.load_mag = raw_load & 0x3FF;
    bool load_cw = (raw_load & 0x400) != 0;
    t.signed_load = load_cw ? -t.load_mag : t.load_mag;

    int spd_mag = raw_speed & 0x3FF;
    bool spd_cw = (raw_speed & 0x400) != 0;
    t.speed_signed = spd_cw ? -spd_mag : spd_mag;

    t.voltage = raw_volt / 10.0;
    return t;
}

std::string makeLoadBar(int load_mag, int threshold) {
    const int width = 20;
    int filled = std::clamp(load_mag * width / 512, 0, width);
    int thresh_pos = std::clamp(threshold * width / 512, 0, width - 1);
    std::string bar(width, '.');
    for (int i = 0; i < filled; i++) bar[i] = '#';
    bar[thresh_pos] = '|'; // penanda ambang batas grasp
    return "[" + bar + "]";
}

void printDetailedStatus() {
    GripperTelemetry t = readTelemetry();
    if (!t.comm_ok) {
        std::cout << "  [ERROR] Gagal membaca telemetri dari Servo ID " << GRIPPER_ID << "!\n";
        return;
    }

    int lag = static_cast<int>(t.present_pos) - static_cast<int>(t.goal_pos);
    double load_pct = (t.load_mag / 1023.0) * 100.0;

    std::cout << "\n---------------- TELEMETRI DEBUG GRIPPER (ID 6) ----------------\n"
              << "  Status Genggam : " << (g_is_holding ? "HOLDING OBJECT" : "IDLE / FREE") << "\n"
              << "  Present Pos    : " << t.present_pos << " ticks ("
              << std::fixed << std::setprecision(2) << rawToRelAngle(t.present_pos)
              << " deg dari titik tutup 0 deg)\n"
              << "  Goal Position  : " << t.goal_pos << " ticks ("
              << rawToRelAngle(t.goal_pos) << " deg)\n"
              << "  Position Lag   : " << std::showpos << lag << " ticks ("
              << ticksToDeg(lag) << " deg)" << std::noshowpos << "\n"
              << "  Present Load   : " << t.load_mag << " / 1023 (" << std::setprecision(1) << load_pct
              << "%, arah=" << (t.signed_load < 0 ? "CW/Tutup" : (t.signed_load > 0 ? "CCW/Buka" : "Netral"))
              << ") " << makeLoadBar(t.load_mag, g_cfg.load_threshold) << "\n"
              << "  Torque Limit   : " << t.torque_limit << " / 1023 ("
              << (t.torque_limit / 1023.0 * 100.0) << "%)\n"
              << "  Voltase & Suhu : " << t.voltage << " V | " << (int)t.temperature << " C\n"
              << "  Hardware Alarm : " << decodeHwError(t.hw_error) << "\n"
              << "----------------------------------------------------------------\n";
}

// ---------------- LIVE DEBUG MONITOR ----------------
void runLiveMonitor(double duration_sec) {
    duration_sec = std::clamp(duration_sec, 1.0, 120.0);
    int steps = static_cast<int>(std::ceil(duration_sec * 10.0)); // 10 Hz

    std::cout << "\n================ LIVE DEBUG MONITOR ID 6 (" << duration_sec << "s @ 10Hz) ================\n";
    std::cout << std::left
              << std::setw(7)  << "Time"
              << std::setw(6)  << "Goal"
              << std::setw(6)  << "Pos"
              << std::setw(9)  << "Sudut"
              << std::setw(6)  << "Lag"
              << std::setw(12) << "Load(Raw/%)"
              << std::setw(24) << "Load Bar (| = Thresh)"
              << std::setw(7)  << "Volt"
              << std::setw(6)  << "Temp"
              << "Status / Event\n";
    std::cout << "---------------------------------------------------------------------------------------------\n";

    for (int i = 0; i < steps; i++) {
        GripperTelemetry t = readTelemetry();
        if (!t.comm_ok) {
            std::cout << "  [COMM ERROR]\n";
            break;
        }

        int lag = static_cast<int>(t.present_pos) - static_cast<int>(t.goal_pos);
        double load_pct = (t.load_mag / 1023.0) * 100.0;
        std::string event = decodeHwError(t.hw_error);

        if (g_is_holding) {
            if (t.load_mag < 35 && std::abs(static_cast<int>(t.present_pos) - getCenterTick()) <= 5) {
                event = "[SLIP / OBJEK LEPAS!]";
                g_is_holding = false;
            } else if (t.load_mag >= g_cfg.load_threshold) {
                event = "HOLDING (SOLID)";
            } else {
                event = "HOLDING (LIGHT)";
            }
        }

        std::ostringstream load_str;
        load_str << std::setw(3) << t.load_mag << " (" << std::fixed << std::setprecision(1) << load_pct << "%)";

        std::cout << std::left
                  << std::setw(7)  << (std::to_string(i / 10) + "." + std::to_string(i % 10) + "s")
                  << std::setw(6)  << t.goal_pos
                  << std::setw(6)  << t.present_pos
                  << std::setw(9)  << std::fixed << std::setprecision(1) << rawToRelAngle(t.present_pos)
                  << std::setw(6)  << std::showpos << lag << std::noshowpos
                  << std::setw(12) << load_str.str()
                  << std::setw(24) << makeLoadBar(t.load_mag, g_cfg.load_threshold)
                  << std::setw(7)  << std::setprecision(1) << t.voltage
                  << std::setw(6)  << (std::to_string(t.temperature) + "C")
                  << event << "\n";

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "=============================================================================================\n";
}

// ---------------- ALGORITMA ADAPTIVE GRASPING ----------------
// Menutup rahang dari posisi terbuka (+deg, tick ~700) turun menuju 0.0 deg (tick 510)
bool executeAdaptiveGrasp(int custom_threshold = -1) {
    int thresh = (custom_threshold > 0) ? std::clamp(custom_threshold, 50, 800) : g_cfg.load_threshold;
    uint8_t err = 0;

    const int closed_tick = getCenterTick(); // 510
    const int max_open_tick = relAngleToTick(MAX_OPEN_DEG); // ~700

    GripperTelemetry init_t = readTelemetry();
    if (!init_t.comm_ok) {
        std::cout << "  [ERROR] Servo ID 6 tidak merespons!\n";
        return false;
    }

    // Nyalakan LED servo saat proses grasping berlangsung
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 1, &err);

    // Atur compliance slope & torsi pencarian
    int search_torque = std::clamp(thresh + 220, 320, 750);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CW_SLOPE,  g_cfg.compliance_slope, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CCW_SLOPE, g_cfg.compliance_slope, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_MOVING_SPEED, g_cfg.closing_speed, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_LIMIT, search_torque, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 1, &err);

    uint16_t start_pos = init_t.present_pos;
    // Jika rahang sedang tertutup rapat (< 10 ticks dari closed_tick 510), buka terlebih dahulu ke +40 deg
    if (static_cast<int>(start_pos) <= closed_tick + 10) {
        uint16_t open_tick = relAngleToTick(40.0);
        std::cout << "  [PRE-OPEN] Rahang sedang tertutup rapat, membuka ke +40.0 deg (tick=" << open_tick << ")...\n";
        packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, open_tick, &err);
        std::this_thread::sleep_for(std::chrono::milliseconds(650));
        start_pos = readTelemetry().present_pos;
    }

    std::cout << "\n================ PROSES ADAPTIVE GRASPING (ID 6) ================\n";
    std::cout << "  Menutup dari tick " << start_pos << " (" << std::fixed << std::setprecision(1) << rawToRelAngle(start_pos)
              << " deg) turun menuju tick " << closed_tick << " (0.0 deg)\n"
              << "  Ambang Load: " << thresh << " | Stall Limit: " << g_cfg.stall_ticks << " ticks\n";
    std::cout << std::left
              << std::setw(6)  << "Step"
              << std::setw(6)  << "Goal"
              << std::setw(6)  << "Pos"
              << std::setw(9)  << "Sudut"
              << std::setw(6)  << "Lag"
              << std::setw(12) << "Load(Raw/%)"
              << std::setw(24) << "Load Bar"
              << "State\n";
    std::cout << "-----------------------------------------------------------------------------\n";

    int goal_tick = std::min<int>(max_open_tick, start_pos);
    int contact_count = 0;
    int step = 0;

    // Langkah penutupan: goal_tick bergerak turun dari nilai tinggi menuju closed_tick (510)
    while (goal_tick > closed_tick) {
        step++;
        goal_tick = std::max(closed_tick, goal_tick - g_cfg.step_ticks);
        packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION,
                                      static_cast<uint16_t>(goal_tick), &err);
        std::this_thread::sleep_for(std::chrono::milliseconds(g_cfg.step_ms));

        GripperTelemetry t = readTelemetry();
        // Ketika menutup ke bawah, jika tertahan objek di atas goal_tick, present_pos > goal_tick
        int lag = static_cast<int>(t.present_pos) - goal_tick;
        double load_pct = (t.load_mag / 1023.0) * 100.0;

        bool trig_load  = (t.load_mag >= thresh);
        bool trig_stall = (lag >= g_cfg.stall_ticks);
        std::string state = "SEARCHING";

        // Abaikan 2 langkah pertama (arus asut motor)
        if (step > 2 && (trig_load || trig_stall)) {
            contact_count++;
            state = "CONTACT (" + std::to_string(contact_count) + "/" + std::to_string(g_cfg.confirm_cycles) + ")";
        } else {
            contact_count = 0;
        }

        std::ostringstream load_str;
        load_str << std::setw(3) << t.load_mag << " (" << std::fixed << std::setprecision(1) << load_pct << "%)";

        std::cout << std::left
                  << std::setw(6)  << step
                  << std::setw(6)  << goal_tick
                  << std::setw(6)  << t.present_pos
                  << std::setw(9)  << std::fixed << std::setprecision(1) << rawToRelAngle(t.present_pos)
                  << std::setw(6)  << std::showpos << lag << std::noshowpos
                  << std::setw(12) << load_str.str()
                  << std::setw(24) << makeLoadBar(t.load_mag, thresh)
                  << state << "\n";

        if (contact_count >= g_cfg.confirm_cycles) {
            // Kunci target pada posisi kontak dikurangi squeeze_ticks (sedikit menekan objek ke arah tutup)
            int hold_tick = std::clamp(static_cast<int>(t.present_pos) - g_cfg.squeeze_ticks,
                                       closed_tick, max_open_tick);
            int hold_torque = std::clamp(t.load_mag + 160, 280, 650);

            packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_LIMIT,
                                          static_cast<uint16_t>(hold_torque), &err);
            packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION,
                                          static_cast<uint16_t>(hold_tick), &err);

            g_is_holding = true;
            g_hold_tick  = static_cast<uint16_t>(hold_tick);

            std::cout << "-----------------------------------------------------------------------------\n";
            std::cout << "  [HOLDING OBJECT] Objek berhasil digenggam!\n"
                      << "    -> Posisi Objek  : tick " << t.present_pos << " ("
                      << std::fixed << std::setprecision(1) << rawToRelAngle(t.present_pos) << " deg)\n"
                      << "    -> Target Kunci  : tick " << hold_tick << " (squeeze -" << g_cfg.squeeze_ticks << " ticks)\n"
                      << "    -> Beban Kontak  : " << t.load_mag << " (" << load_pct << "%)\n"
                      << "    -> Holding Torque: " << hold_torque << " / 1023 ("
                      << (hold_torque / 1023.0 * 100.0) << "%)\n";
            std::cout << "=============================================================================\n";
            return true;
        }
    }

    g_is_holding = false;
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 0, &err);
    std::cout << "-----------------------------------------------------------------------------\n";
    std::cout << "  [CLOSED EMPTY] Rahang mencapai 0.0 deg (tick=" << closed_tick
              << ") tanpa mendeteksi objek.\n";
    std::cout << "=============================================================================\n";
    return false;
}

void moveGripperDeg(double deg, int speed = 80) {
    uint8_t err = 0;
    uint16_t goal = relAngleToTick(deg);
    g_is_holding = false;

    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 0, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CW_SLOPE,  g_cfg.compliance_slope, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_CCW_SLOPE, g_cfg.compliance_slope, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_MOVING_SPEED, speed, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_LIMIT, 600, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 1, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, goal, &err);

    std::cout << "  [MOVE] Gripper ID 6 menuju " << std::fixed << std::setprecision(1)
              << std::clamp(deg, 0.0, MAX_OPEN_DEG) << " deg (tick=" << goal << ")\n";
}

void printMenu() {
    std::cout << "\n================ ADAPTIVE GRIPPER & DEBUG MONITOR (ID 6) ================\n"
              << "  Perintah Kontrol:\n"
              << "    grasp [load_thresh]   -> Jepit objek secara adaptif (default load=" << g_cfg.load_threshold << ")\n"
              << "    release [deg]         -> Buka rahang gripper (default +45.0 deg, misal: release atau release 50)\n"
              << "    move <deg>            -> Gerakkan ke sudut tertentu (0.0=tutup rapat s/d +55.0=buka penuh)\n"
              << "    free                  -> Matikan torsi ID 6\n"
              << "    lock                  -> Kunci posisi ID 6 saat ini\n\n"
              << "  Perintah Debug & Monitoring:\n"
              << "    status                -> Tampilkan laporan telemetri lengkap ID 6 saat ini\n"
              << "    monitor [detik]       -> Live debug stream 10 Hz selama [detik] (default 10s)\n"
              << "    config <load> <squeeze> <speed> <slope>\n"
              << "                          -> Ubah parameter adaptive grasp (saat ini: load=" << g_cfg.load_threshold
              << ", squeeze=" << g_cfg.squeeze_ticks << ", speed=" << g_cfg.closing_speed
              << ", slope=" << (int)g_cfg.compliance_slope << ")\n"
              << "    help                  -> Tampilkan menu ini\n"
              << "    quit                  -> Matikan torsi & keluar\n"
              << "=========================================================================\n";
}

// ---------------- MAIN ----------------
int main() {
    portHandler   = dynamixel::PortHandler::getPortHandler(DEVICENAME);
    packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

    if (!portHandler->openPort() || !portHandler->setBaudRate(BAUDRATE)) {
        std::cerr << "[ERROR] Gagal membuka port " << DEVICENAME << " @ " << BAUDRATE << " bps!\n";
        return -1;
    }

    uint16_t model = 0;
    uint8_t err = 0;
    if (packetHandler->ping(portHandler, GRIPPER_ID, &model, &err) != COMM_SUCCESS) {
        std::cerr << "[ERROR] Gripper Servo ID " << GRIPPER_ID << " tidak terdeteksi!\n";
        portHandler->closePort();
        return -1;
    }

    // Terapkan batas aman CW/CCW dari calibration_data.h (490 - 750)
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 0, &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_CW_ANGLE_LIMIT,  getCwLimit(),  &err);
    packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_CCW_ANGLE_LIMIT, getCcwLimit(), &err);

    std::cout << "[OK] Gripper ID 6 terdeteksi (Model: " << model
              << ", Center=" << getCenterTick()
              << ", CW_Lim=" << getCwLimit()
              << ", CCW_Lim=" << getCcwLimit() << ")\n";

    printDetailedStatus();
    printMenu();

    while (true) {
        std::cout << "\nGripper(ID6)> ";
        std::string line;
        if (!std::getline(std::cin, line)) break;

        std::stringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd.empty()) continue;

        if (cmd == "grasp") {
            int thresh = -1;
            ss >> thresh;
            executeAdaptiveGrasp(thresh);
        } else if (cmd == "release") {
            double deg = DEFAULT_RELEASE_DEG;
            ss >> deg;
            moveGripperDeg(deg);
        } else if (cmd == "move") {
            double deg = 0.0;
            if (ss >> deg) {
                moveGripperDeg(deg);
            } else {
                std::cout << "  [!] Format: move <deg> (rentang 0.0 s/d +55.0 deg)\n";
            }
        } else if (cmd == "free") {
            g_is_holding = false;
            packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 0, &err);
            packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 0, &err);
            std::cout << "  [FREE] Torsi ID 6 dimatikan.\n";
        } else if (cmd == "lock") {
            GripperTelemetry t = readTelemetry();
            if (t.comm_ok) {
                packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, t.present_pos, &err);
                packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 1, &err);
                std::cout << "  [LOCK] ID 6 dikunci pada tick " << t.present_pos
                          << " (" << rawToRelAngle(t.present_pos) << " deg)\n";
            }
        } else if (cmd == "status") {
            printDetailedStatus();
        } else if (cmd == "monitor") {
            double secs = 10.0;
            ss >> secs;
            runLiveMonitor(secs);
        } else if (cmd == "config") {
            int load, squeeze, speed, slope;
            if (ss >> load >> squeeze >> speed >> slope) {
                g_cfg.load_threshold   = std::clamp(load, 50, 800);
                g_cfg.squeeze_ticks    = std::clamp(squeeze, 0, 30);
                g_cfg.closing_speed    = std::clamp(speed, 10, 400);
                g_cfg.compliance_slope = static_cast<uint8_t>(std::clamp(slope, 8, 254));
                std::cout << "  [CONFIG] Diperbarui: load=" << g_cfg.load_threshold
                          << ", squeeze=" << g_cfg.squeeze_ticks
                          << ", speed=" << g_cfg.closing_speed
                          << ", slope=" << (int)g_cfg.compliance_slope << "\n";
            } else {
                std::cout << "  [!] Format: config <load_thresh> <squeeze_ticks> <speed> <slope>\n"
                          << "      Contoh: config 180 5 60 64\n";
            }
        } else if (cmd == "help") {
            printMenu();
        } else if (cmd == "quit" || cmd == "exit" || cmd == "q") {
            break;
        } else {
            std::cout << "  [!] Perintah tidak dikenal. Ketik 'help' untuk bantuan.\n";
        }
    }

    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_LED, 0, &err);
    packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 0, &err);
    portHandler->closePort();
    std::cout << "Torsi ID 6 dimatikan dan port ditutup.\n";
    return 0;
}
