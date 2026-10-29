#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>
#include <filesystem>
#include <map>
#include <string>
#include <sstream>
#include <dynamixel_sdk.h>

// ---------------- REGISTER AX-18A (PROTOCOL 1.0) ----------------
#define ADDR_MODEL_NUMBER       0
#define ADDR_FIRMWARE           2
#define ADDR_CW_ANGLE_LIMIT     6   // EEPROM (2 byte, 0-1023)
#define ADDR_CCW_ANGLE_LIMIT    8   // EEPROM (2 byte, 0-1023)
#define ADDR_TORQUE_ENABLE      24  // RAM (1 byte, 0/1)
#define ADDR_CW_SLOPE           28  // RAM (1 byte)
#define ADDR_CCW_SLOPE          29  // RAM (1 byte)
#define ADDR_GOAL_POSITION      30  // RAM (2 byte, 0-1023)
#define ADDR_MOVING_SPEED       32  // RAM (2 byte, 0-1023)
#define ADDR_TORQUE_LIMIT       34  // RAM (2 byte, 0-1023)
#define ADDR_PRESENT_POS        36  // RAM (2 byte, 0-1023)
#define ADDR_PRESENT_LOAD       40  // RAM (2 byte, bit 10 = arah, 0-9 = magnitude)
#define ADDR_PRESENT_VOLT       42  // RAM (1 byte, x0.1V)
#define ADDR_PRESENT_TEMP       43  // RAM (1 byte, deg C)

#define LEN_GOAL_POSITION       2
#define PROTOCOL_VERSION        1.0
#define BAUDRATE                1000000
#define DEVICENAME              "/dev/ttyUSB0"

#define MAX_ANGLE               300.0
#define MAX_TICKS               1023
#define SAFE_MIN_TICK           10      // ~2.9 deg (mencegah masuk dead zone bawah 0)
#define SAFE_MAX_TICK           1010    // ~296.2 deg (mencegah masuk dead zone atas 1011-1023)
#define NOMINAL_CENTER_TICK     512     // 150.0 derajat pada Dynamixel Wizard
#define COMPLIANCE_SLOPE        32

// Parameter keamanan saat kalibrasi & uji awal
#define LOW_SPEED_VAL           40      // ~4.5 RPM (sangat lambat & aman)
#define LOW_TORQUE_VAL          600     // ~58% torsi maks agar kuat mengangkat lengan melawan gravitasi
#define INITIAL_SAFE_DEG        15.0    // Batas konservatif awal (+/- 15 derajat)
#define FULL_WORK_RANGE_DEG     260.0   // Rentang kerja maksimum penuh (J1 & J2)
#define FULL_GRIP_OPEN_DEG      80.0    // Rentang buka penuh Gripper ID 2 (~780 ticks)

const uint8_t SERVO_IDS[] = {1, 4, 5, 2};

struct ServoCalib {
    uint8_t  id;
    uint16_t raw_zero;      // Posisi aktual saat lengan di titik 0 mekanis
    int16_t  trim_ticks;    // Koreksi halus (khususnya ID 4 & 5)
    uint16_t cw_limit;      // Batas minimum tick (CW)
    uint16_t ccw_limit;     // Batas maksimum tick (CCW)

    uint16_t effectiveCenter() const {
        int val = static_cast<int>(raw_zero) + trim_ticks;
        return static_cast<uint16_t>(std::clamp(val, 0, MAX_TICKS));
    }
};

dynamixel::PortHandler   *portHandler   = nullptr;
dynamixel::PacketHandler *packetHandler = nullptr;

// ---------------- KONVERSI & HELPER ----------------
double ticksToDeg(int ticks) {
    return ticks * (MAX_ANGLE / MAX_TICKS);
}

int degToTicks(double deg) {
    return static_cast<int>(std::round(deg / MAX_ANGLE * MAX_TICKS));
}

uint16_t clampTicks(int ticks) {
    return static_cast<uint16_t>(std::clamp(ticks, SAFE_MIN_TICK, SAFE_MAX_TICK));
}

bool write1Byte(uint8_t id, uint16_t addr, uint8_t val) {
    uint8_t err = 0;
    int res = packetHandler->write1ByteTxRx(portHandler, id, addr, val, &err);
    return (res == COMM_SUCCESS && err == 0);
}

bool write2Byte(uint8_t id, uint16_t addr, uint16_t val) {
    uint8_t err = 0;
    int res = packetHandler->write2ByteTxRx(portHandler, id, addr, val, &err);
    return (res == COMM_SUCCESS && err == 0);
}

bool read2Byte(uint8_t id, uint16_t addr, uint16_t &val) {
    uint8_t err = 0;
    int res = packetHandler->read2ByteTxRx(portHandler, id, addr, &val, &err);
    return (res == COMM_SUCCESS);
}

void setTorqueAll(bool enable) {
    if (!enable) {
        // Pastikan servo non-aktif (misal ID 1 & 2) juga dimatikan torsinya jika terhubung
        const uint8_t all_possible[] = {1, 4, 5, 2};
        for (uint8_t id : all_possible) {
            write1Byte(id, ADDR_TORQUE_ENABLE, 0);
        }
        return;
    }
    for (uint8_t id : SERVO_IDS) {
        write1Byte(id, ADDR_TORQUE_ENABLE, 1);
    }
}

void setLowSpeedAndTorque(uint16_t speed = LOW_SPEED_VAL, uint16_t torque = LOW_TORQUE_VAL) {
    for (uint8_t id : SERVO_IDS) {
        write1Byte(id, ADDR_CW_SLOPE,     COMPLIANCE_SLOPE);
        write1Byte(id, ADDR_CCW_SLOPE,    COMPLIANCE_SLOPE);
        write2Byte(id, ADDR_MOVING_SPEED, speed);
        write2Byte(id, ADDR_TORQUE_LIMIT, torque);
    }
}

// Membaca beban (Present Load): negatif = CW, positif = CCW
int readSignedLoad(uint8_t id) {
    uint16_t raw = 0;
    if (!read2Byte(id, ADDR_PRESENT_LOAD, raw)) return 0;
    int mag = raw & 0x3FF;
    bool cw = (raw & 0x400) != 0;
    return cw ? -mag : mag;
}

void waitEnter(const std::string &prompt) {
    std::cout << prompt << " [Tekan ENTER untuk lanjut] ";
    std::string dummy;
    std::getline(std::cin, dummy);
}

// ---------------- VERIFIKASI KONEKSI ----------------
bool checkServos() {
    std::cout << "Memeriksa koneksi servo ID 1, 4, 5, 2 pada " << DEVICENAME
              << " @ " << BAUDRATE << " bps...\n";
    bool all_ok = true;
    for (uint8_t id : SERVO_IDS) {
        uint16_t model = 0;
        uint8_t err = 0;
        if (packetHandler->ping(portHandler, id, &model, &err) != COMM_SUCCESS) {
            std::cerr << "  [ERROR] Servo ID " << (int)id << " tidak merespons!\n";
            all_ok = false;
        } else {
            std::cout << "  [OK] Servo ID " << (int)id << " terdeteksi (Model: " << model << ")\n";
        }
    }
    return all_ok;
}

// ---------------- STEP 1 & 2: TORQUE OFF & POSISI NOL MEKANIS ----------------
void step1And2_DisableTorqueAndPlaceZero() {
    std::cout << "\n========================================================\n";
    std::cout << "STEP 1 & 2: DISABLE TORQUE & POSISIKAN LENGAN DI TITIK 0\n";
    std::cout << "========================================================\n";
    setTorqueAll(false);
    std::cout << "[INFO] Torque pada ID 1, 4, 5, 2 telah DIMATIKAN.\n";
    std::cout << "[INSTRUKSI] Posisikan lengan secara mekanis pada posisi 0 (Zero Position) yang diinginkan.\n";
    std::cout << "            Jangan hanya mengandalkan nilai tengah 512 (150 deg) Dynamixel Wizard.\n\n";

    while (true) {
        std::cout << "Ketik 'p' lalu ENTER untuk preview posisi saat ini, atau langsung ENTER jika posisi 0 sudah siap: ";
        std::string line;
        std::getline(std::cin, line);
        if (line == "p" || line == "P") {
            std::cout << "  Posisi saat ini:\n";
            for (uint8_t id : SERVO_IDS) {
                uint16_t pos = 0;
                read2Byte(id, ADDR_PRESENT_POS, pos);
                std::cout << "    ID " << (int)id << ": " << std::setw(4) << pos << " ticks ("
                          << std::fixed << std::setprecision(2) << ticksToDeg(pos) << " deg)\n";
            }
        } else {
            break;
        }
    }
}

// ---------------- STEP 3 & 4: BACA & REKAM CENTER AKTUAL ----------------
void step3And4_ReadAndRecordCenters(std::map<uint8_t, ServoCalib> &calib) {
    std::cout << "\n========================================================\n";
    std::cout << "STEP 3 & 4: BACA & REKAM CENTER AKTUAL (ID 1, 4, 5, 2)\n";
    std::cout << "========================================================\n";
    std::cout << std::left
              << std::setw(6)  << "ID"
              << std::setw(12) << "Raw Tick"
              << std::setw(14) << "Sudut (deg)"
              << std::setw(22) << "Selisih vs Wizard(512)" << "\n";
    std::cout << "--------------------------------------------------------\n";

    for (uint8_t id : SERVO_IDS) {
        uint16_t pos = NOMINAL_CENTER_TICK;
        read2Byte(id, ADDR_PRESENT_POS, pos);
        calib[id] = ServoCalib{id, pos, 0, 0, MAX_TICKS};

        int diff_ticks = static_cast<int>(pos) - NOMINAL_CENTER_TICK;
        std::cout << std::setw(6)  << (int)id
                  << std::setw(12) << pos
                  << std::setw(14) << std::fixed << std::setprecision(2) << ticksToDeg(pos)
                  << std::showpos << diff_ticks << " ticks ("
                  << ticksToDeg(diff_ticks) << " deg)" << std::noshowpos << "\n";
    }
}

// ---------------- STEP 5: KOREKSI HALUS (FINE-TUNE) ID 4 & 5 ----------------
void step5_AlignPairedJoint4And5(std::map<uint8_t, ServoCalib> &calib) {
    std::cout << "\n========================================================\n";
    std::cout << "STEP 5: KOREKSI HALUS (ALIGNMENT) JOINT PARAREL ID 4 & 5\n";
    std::cout << "========================================================\n";
    std::cout << "Mengunci ID 4 & 5 pada posisi center aktual dengan torsi & kecepatan rendah...\n";

    // Buka limit sementara agar tidak tertahan limit lama di EEPROM
    write1Byte(4, ADDR_TORQUE_ENABLE, 0);
    write1Byte(5, ADDR_TORQUE_ENABLE, 0);
    write2Byte(4, ADDR_CW_ANGLE_LIMIT, 0);
    write2Byte(4, ADDR_CCW_ANGLE_LIMIT, MAX_TICKS);
    write2Byte(5, ADDR_CW_ANGLE_LIMIT, 0);
    write2Byte(5, ADDR_CCW_ANGLE_LIMIT, MAX_TICKS);

    setLowSpeedAndTorque(LOW_SPEED_VAL, LOW_TORQUE_VAL);
    write2Byte(4, ADDR_GOAL_POSITION, calib[4].effectiveCenter());
    write2Byte(5, ADDR_GOAL_POSITION, calib[5].effectiveCenter());
    write1Byte(4, ADDR_TORQUE_ENABLE, 1);
    write1Byte(5, ADDR_TORQUE_ENABLE, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    std::cout << "\nGunakan perintah berikut untuk menyelaraskan ID 4 & 5 agar tidak saling melawan:\n"
              << "  4 +n / 4 -n  -> Geser trim ID 4 sebanyak +n atau -n ticks (contoh: 4 +2 atau 4 -1)\n"
              << "  5 +n / 5 -n  -> Geser trim ID 5 sebanyak +n atau -n ticks (contoh: 5 -2 atau 5 +1)\n"
              << "  r            -> Refresh pembacaan posisi & beban (load)\n"
              << "  ok           -> Selesai koreksi ID 4 & 5\n\n";

    while (true) {
        uint16_t p4 = 0, p5 = 0;
        read2Byte(4, ADDR_PRESENT_POS, p4);
        read2Byte(5, ADDR_PRESENT_POS, p5);
        int load4 = readSignedLoad(4);
        int load5 = readSignedLoad(5);

        std::cout << "  [Status] ID 4: center=" << calib[4].effectiveCenter()
                  << " (trim=" << std::showpos << calib[4].trim_ticks << std::noshowpos
                  << ", pos=" << p4 << ", load=" << load4 << ") | "
                  << "ID 5: center=" << calib[5].effectiveCenter()
                  << " (trim=" << std::showpos << calib[5].trim_ticks << std::noshowpos
                  << ", pos=" << p5 << ", load=" << load5 << ")\n";
        std::cout << "Perintah koreksi (4 +/-n | 5 +/-n | r | ok): ";

        std::string line;
        if (!std::getline(std::cin, line) || line == "ok" || line == "OK") break;
        if (line.empty() || line == "r" || line == "R") continue;

        std::stringstream ss(line);
        int id = 0, delta = 0;
        if ((ss >> id >> delta) && (id == 4 || id == 5)) {
            calib[id].trim_ticks += delta;
            uint16_t target = calib[id].effectiveCenter();
            write2Byte(id, ADDR_GOAL_POSITION, target);
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        } else {
            std::cout << "  [!] Format tidak dikenali. Contoh: '4 +1', '5 -2', 'r', atau 'ok'.\n";
        }
    }
}

// ---------------- STEP 6: SET CONSERVATIVE CW/CCW LIMITS ----------------
void applyAngleLimitsDeg(std::map<uint8_t, ServoCalib> &calib,
                         double j1_deg, double j2_deg, double grip_deg) {
    // Matikan torsi sebentar karena CW/CCW Angle Limit berada di area EEPROM
    setTorqueAll(false);

    int d1 = degToTicks(j1_deg);
    int d2 = degToTicks(j2_deg);
    int dg = degToTicks(grip_deg);

    // Tetapkan batas sesuai geometri kerja tiap joint dengan proteksi SAFE_MIN_TICK..SAFE_MAX_TICK (10..1010)
    // agar tidak pernah melewati 1011-1023 (Zona Buta)
    if (calib.count(4)) {
        calib[4].cw_limit  = clampTicks(calib[4].effectiveCenter() - d1);
        calib[4].ccw_limit = clampTicks(calib[4].effectiveCenter() + d1);
    }
    if (calib.count(5)) {
        calib[5].cw_limit  = clampTicks(calib[5].effectiveCenter() - d1);
        calib[5].ccw_limit = clampTicks(calib[5].effectiveCenter() + d1);
    }
    if (calib.count(1)) {
        calib[1].cw_limit  = clampTicks(calib[1].effectiveCenter() - d2);
        calib[1].ccw_limit = clampTicks(calib[1].effectiveCenter() + d2);
    }
    if (calib.count(2)) {
        // ID 2 (Gripper): center (~506) adalah posisi menutup rapat, arah membuka adalah +ticks (hingga ~753+)
        calib[2].cw_limit  = clampTicks(calib[2].effectiveCenter() - 15);
        calib[2].ccw_limit = clampTicks(calib[2].effectiveCenter() + dg);
    }

    std::cout << "\nMenulis batas CW/CCW ke EEPROM:\n";
    for (uint8_t id : SERVO_IDS) {
        write2Byte(id, ADDR_CW_ANGLE_LIMIT,  calib[id].cw_limit);
        write2Byte(id, ADDR_CCW_ANGLE_LIMIT, calib[id].ccw_limit);
        std::cout << "  ID " << (int)id
                  << " -> CW Limit: "  << std::setw(4) << calib[id].cw_limit
                  << " (" << std::fixed << std::setprecision(1) << ticksToDeg(calib[id].cw_limit) << " deg)"
                  << " | CCW Limit: " << std::setw(4) << calib[id].ccw_limit
                  << " (" << ticksToDeg(calib[id].ccw_limit) << " deg)\n";
    }

    // Kembalikan konfigurasi low speed & low torque lalu aktifkan torsi
    setLowSpeedAndTorque(LOW_SPEED_VAL, LOW_TORQUE_VAL);
    for (uint8_t id : SERVO_IDS) {
        uint16_t cur = calib[id].effectiveCenter();
        read2Byte(id, ADDR_PRESENT_POS, cur);
        cur = std::clamp(cur, calib[id].cw_limit, calib[id].ccw_limit);
        write2Byte(id, ADDR_GOAL_POSITION, cur);
        write1Byte(id, ADDR_TORQUE_ENABLE, 1);
    }
}

// ---------------- GERAK JOINT TERKALIBRASI ----------------
// Menggerakkan Joint 1 (ID 4 & ID 5 berlawanan arah secara sinkron)
void moveJoint1Rel(dynamixel::GroupSyncWrite &gsw,
                   const std::map<uint8_t, ServoCalib> &calib,
                   double delta_deg) {
    if (!calib.count(4) || !calib.count(5)) {
        std::cout << "  [!] Servo ID 4 & 5 tidak aktif.\n";
        return;
    }
    int dt = degToTicks(delta_deg);
    uint16_t g4 = std::clamp(clampTicks(calib.at(4).effectiveCenter() + dt),
                             calib.at(4).cw_limit, calib.at(4).ccw_limit);
    uint16_t g5 = std::clamp(clampTicks(calib.at(5).effectiveCenter() - dt),
                             calib.at(5).cw_limit, calib.at(5).ccw_limit);

    gsw.clearParam();
    uint8_t p4[2] = { DXL_LOBYTE(g4), DXL_HIBYTE(g4) };
    uint8_t p5[2] = { DXL_LOBYTE(g5), DXL_HIBYTE(g5) };
    gsw.addParam(4, p4);
    gsw.addParam(5, p5);
    gsw.txPacket();

    std::cout << "  -> Joint 1 menuju " << std::showpos << delta_deg << std::noshowpos
              << " deg (ID4=" << g4 << ", ID5=" << g5 << ")\n";
}

// Menggerakkan servo tunggal (ID 1 atau ID 2) relatif terhadap center-nya
// delta_deg positif = CCW (+ticks), delta_deg negatif = CW (-ticks)
void moveSingleRel(const std::map<uint8_t, ServoCalib> &calib,
                   uint8_t id, double delta_deg) {
    if (!calib.count(id)) {
        std::cout << "  [!] Servo ID " << (int)id << " sedang dinonaktifkan.\n";
        return;
    }
    int dt = degToTicks(delta_deg);
    uint16_t goal = std::clamp(clampTicks(calib.at(id).effectiveCenter() + dt),
                               calib.at(id).cw_limit, calib.at(id).ccw_limit);
    write2Byte(id, ADDR_GOAL_POSITION, goal);
    std::cout << "  -> Servo ID " << (int)id << " menuju " << std::showpos << delta_deg
              << std::noshowpos << " deg dari center (tick=" << goal << ")\n";
}

// ---------------- STEP 7 & 8: UJI GERAK & EKSPANSI RENTANG BERTAHAP ----------------
void step7And8_TestAndExpandRange(dynamixel::GroupSyncWrite &gsw,
                                  std::map<uint8_t, ServoCalib> &calib) {
    double max_j1   = INITIAL_SAFE_DEG;
    double max_j2   = INITIAL_SAFE_DEG;
    double max_grip = INITIAL_SAFE_DEG;

    std::cout << "\n========================================================\n";
    std::cout << "STEP 6: SET BATAS KONSERVATIF AWAL (+/- " << INITIAL_SAFE_DEG << " DERAJAT)\n";
    std::cout << "========================================================\n";
    applyAngleLimitsDeg(calib, max_j1, max_j2, max_grip);

    std::cout << "\n========================================================\n";
    std::cout << "STEP 7 & 8: UJI KECEPATAN/TORSI RENDAH & EKSPANSI RENTANG\n";
    std::cout << "========================================================\n";

    while (true) {
        std::cout << "\n--- Menu Uji & Ekspansi Rentang (Speed=" << LOW_SPEED_VAL
                  << ", TorqueLimit=" << LOW_TORQUE_VAL << ") ---\n"
                  << "  Batas aktif saat ini : J1(ID4&5)=+/-" << max_j1 << " deg | "
                  << "J2(ID1)=+/-" << max_j2 << " deg | Gripper(ID2)=0..+" << max_grip << " deg\n"
                  << "  Perintah:\n"
                  << "    j1 <deg>     -> Uji Joint 1 (+/-deg, misal: j1 -10, j1 -90, atau j1 0)\n"
                  << "    j2 <deg>     -> Uji Joint 2 (ID 1, -deg=angkat ke atas, +deg=tekuk dalam)\n"
                  << "    grip <deg>   -> Uji Gripper (ID 2, 0=tutup rapat s/d +75=buka penuh)\n"
                  << "    zero         -> Kembalikan semua servo aktif ke posisi 0 terkalibrasi\n"
                  << "    expand <d1> [d2] [dg] -> Perlebar batas derajat (maks 260), misal: expand 260 260 80\n"
                  << "    status       -> Baca posisi & beban (load) semua servo aktif saat ini\n"
                  << "    save         -> Terapkan rentang kerja penuh, simpan calibration_data.h & keluar\n"
                  << "Pilih perintah: ";

        std::string line;
        if (!std::getline(std::cin, line)) break;
        std::stringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "j1") {
            double deg = 0;
            if (ss >> deg) {
                deg = std::clamp(deg, -max_j1, max_j1);
                moveJoint1Rel(gsw, calib, deg);
            }
        } else if (cmd == "j2") {
            double deg = 0;
            if (ss >> deg) {
                deg = std::clamp(deg, -max_j2, max_j2);
                moveSingleRel(calib, 1, deg);
            }
        } else if (cmd == "grip") {
            double deg = 0;
            if (ss >> deg) {
                deg = std::clamp(deg, 0.0, max_grip);
                moveSingleRel(calib, 2, deg);
            }
        } else if (cmd == "zero") {
            if (calib.count(4) && calib.count(5)) moveJoint1Rel(gsw, calib, 0.0);
            if (calib.count(1)) moveSingleRel(calib, 1, 0.0);
            if (calib.count(2)) moveSingleRel(calib, 2, 0.0);
        } else if (cmd == "expand") {
            double d1 = max_j1, d2 = max_j2, dg = max_grip;
            if (ss >> d1) {
                if (!(ss >> d2 >> dg)) {
                    d2 = d1;
                    dg = std::min(d1, FULL_GRIP_OPEN_DEG);
                }
                max_j1   = std::clamp(d1, 5.0, FULL_WORK_RANGE_DEG);
                max_j2   = std::clamp(d2, 5.0, FULL_WORK_RANGE_DEG);
                max_grip = std::clamp(dg, 5.0, FULL_GRIP_OPEN_DEG);
                applyAngleLimitsDeg(calib, max_j1, max_j2, max_grip);
            } else {
                std::cout << "  [!] Format: expand <deg> atau expand <max_j1> <max_j2> <max_grip>\n";
            }
        } else if (cmd == "status") {
            for (uint8_t id : SERVO_IDS) {
                uint16_t pos = 0;
                read2Byte(id, ADDR_PRESENT_POS, pos);
                int load = readSignedLoad(id);
                double rel_deg = ticksToDeg(static_cast<int>(pos) - calib[id].effectiveCenter());
                std::cout << "  ID " << (int)id
                          << ": pos=" << std::setw(4) << pos
                          << " (" << std::showpos << std::fixed << std::setprecision(2)
                          << rel_deg << " deg dari center" << std::noshowpos
                          << "), load=" << load << "\n";
            }
        } else if (cmd == "save") {
            // Pastikan saat save dilakukan, batas kerja di EEPROM & calibration_data.h
            // dibuka ke rentang operasional penuh (bukan tertinggal di limit konservatif +/-15 deg)
            if (max_j1 < FULL_WORK_RANGE_DEG || max_j2 < FULL_WORK_RANGE_DEG || max_grip < FULL_GRIP_OPEN_DEG) {
                std::cout << "\n[INFO] Membuka CW/CCW limit ke rentang kerja maksimum yang aman sebelum menyimpan...\n";
                max_j1   = FULL_WORK_RANGE_DEG;
                max_j2   = FULL_WORK_RANGE_DEG;
                max_grip = FULL_GRIP_OPEN_DEG;
                applyAngleLimitsDeg(calib, max_j1, max_j2, max_grip);
            }
            break;
        } else if (!cmd.empty()) {
            std::cout << "  [!] Perintah tidak dikenal.\n";
        }
    }
}

// ---------------- SIMPAN HEADER KALIBRASI ----------------
void saveCalibrationHeader(const std::map<uint8_t, ServoCalib> &calib, const std::string &path) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "[ERROR] Gagal menulis file " << path << "\n";
        return;
    }

    out << "#pragma once\n"
        << "#include <cstdint>\n\n"
        << "// Dihasilkan otomatis oleh calibrate_arm.cpp\n"
        << "struct CalibratedServoConfig {\n"
        << "    uint8_t  id;\n"
        << "    uint16_t raw_zero;\n"
        << "    int16_t  trim_ticks;\n"
        << "    uint16_t center_tick;\n"
        << "    uint16_t cw_limit;\n"
        << "    uint16_t ccw_limit;\n"
        << "};\n\n"
        << "constexpr CalibratedServoConfig ARM_CALIBRATION[] = {\n";

    for (uint8_t id : SERVO_IDS) {
        const auto &c = calib.at(id);
        out << "    { " << (int)c.id << ", "
            << c.raw_zero << ", "
            << c.trim_ticks << ", "
            << c.effectiveCenter() << ", "
            << c.cw_limit << ", "
            << c.ccw_limit << " },\n";
    }
    out << "};\n";
    out.close();
    std::cout << "\n[SUKSES] Data kalibrasi disimpan ke: " << path << "\n";
}

// ---------------- MAIN ----------------
int main() {
    portHandler   = dynamixel::PortHandler::getPortHandler(DEVICENAME);
    packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);
    dynamixel::GroupSyncWrite gsw(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION);

    if (!portHandler->openPort() || !portHandler->setBaudRate(BAUDRATE)) {
        std::cerr << "[ERROR] Gagal membuka port " << DEVICENAME << " atau set baudrate!\n";
        return -1;
    }

    if (!checkServos()) {
        portHandler->closePort();
        return -1;
    }

    std::map<uint8_t, ServoCalib> calib;

    // Step 1 & 2: Disable torque & posisikan arm secara mekanis ke titik 0
    step1And2_DisableTorqueAndPlaceZero();

    // Step 3 & 4: Baca & catat center aktual untuk ID 1, 4, 5, dan 2
    step3And4_ReadAndRecordCenters(calib);

    // Step 5: Koreksi halus (fine-tuning) untuk ID 4 dan 5 hingga sejajar & tidak saling melawan
    step5_AlignPairedJoint4And5(calib);

    // Step 6, 7, 8: Set limit konservatif, uji pada speed/torsi rendah, dan ekspansi bertahap
    step7And8_TestAndExpandRange(gsw, calib);

    // Simpan hasil ke header (otomatis mendeteksi apakah dijalankan dari root atau subfolder)
    std::string header_path = std::filesystem::exists("opencm_dynamixel_servos/calibrate_arm.cpp")
                                  ? "opencm_dynamixel_servos/calibration_data.h"
                                  : "calibration_data.h";
    saveCalibrationHeader(calib, header_path);

    // Matikan torsi dengan aman sebelum keluar
    setTorqueAll(false);
    portHandler->closePort();
    std::cout << "Torque dimatikan dan port ditutup.\n";
    return 0;
}