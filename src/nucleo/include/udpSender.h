#ifndef UDP_SENDER_HEADER
#define UDP_SENDER_HEADER

#include <arpa/inet.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <cstddef>

class udpSender {
 private:
  sockaddr_in addr{};
  int _socket{-1};

 public:
  udpSender() = default;
  udpSender(const char *__ip, int __port, int __socket);
  udpSender(const udpSender &) = default;
  udpSender &operator=(const udpSender &) = default;

  void send(const void *__buff, size_t __len) const;

  template <typename T>
  void put(const T *__buff) const {
    send(__buff, sizeof(*__buff));
  }
};

#endif
