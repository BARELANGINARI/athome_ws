#include "udpSender.h"

udpSender::udpSender(const char *__ip, int __port, int __socket) {
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(__port);
  addr.sin_addr.s_addr = inet_addr(__ip);

  _socket = __socket;
}

void udpSender::send(const void *__buff, size_t __len) const {
  if (_socket < 0 || __buff == nullptr || __len == 0) return;
  sendto(_socket, __buff, __len, 0, (struct sockaddr *)&addr, sizeof(addr));
}
