#ifndef UDP_SERVER_H
#define UDP_SERVER_H

#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "udpSender.h"

#pragma pack(1)

struct STM32toPC {
  float batrePC, batreMotor;
  float Yaw, AccelX, AccelY;
  int EKiri, Ekanan, Ebelakang;
  int EDKiri, EDKanan;
  int PotKiri, PotKanan;
  uint8_t kicked;
  uint8_t lifterState;
  uint8_t proxy;
  uint8_t PushButton;
  uint8_t capacitorFull;
};

struct PCtoSTM32 {
  int16_t MKiri{0}, MKanan{0}, MBelakang{0};
  int16_t DKiri{0}, DKanan{0};
  uint8_t lifter_target{0};
  uint8_t kickPower{0};
};

static_assert(sizeof(PCtoSTM32) == 12,
              "PCtoSTM32 must be 12 bytes (motor + dribbler + lifter + kicker)");

#pragma pack()

class udpServer {
 private:
  struct receiveHelper {
    receiveHelper(void *buff, size_t __len, const sockaddr_in &addr,
                  std::function<void()> cb)
        : buffPtr(buff), len(__len), callback(std::move(cb)) {
      port = addr.sin_port;
    }

    void *buffPtr;
    size_t len;
    std::function<void()> callback;
    in_port_t port;
  };

  int _socket{-1};
  int _port{0};
  std::vector<receiveHelper> selector;
  mutable std::mutex _mtx;
  std::thread _rx_thread;
  std::atomic<bool> _running{false};

  void receiverLoop();
  void recvMessage();

 public:
  udpServer() = default;
  udpServer(int port, const std::string &local_ip = "");
  ~udpServer();

  udpServer(const udpServer &) = delete;
  udpServer &operator=(const udpServer &) = delete;

  void connectServer(int port, const std::string &local_ip = "");
  udpSender addSender(const std::string &ip, int port);
  void addReceiver(const std::string &ip, int port, void *buffPtr, size_t len,
                   std::function<void()> cb = nullptr);

  void receiverStart();
  void receiverStop();
};

#endif
