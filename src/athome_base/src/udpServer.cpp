#include "udpServer.h"

udpServer::udpServer(int port, const std::string &local_ip) {
  connectServer(port, local_ip);
}

udpServer::~udpServer() { receiverStop(); }

void udpServer::connectServer(int port, const std::string &local_ip) {
  _port = port;

  _socket = socket(AF_INET, SOCK_DGRAM, 0);
  if (_socket < 0) {
    perror("udpServer socket");
    return;
  }

  int opt = 1;
  setsockopt(_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in servaddr;
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(_port);
  if (local_ip.empty())
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  else
    servaddr.sin_addr.s_addr = inet_addr(local_ip.c_str());

  if (bind(_socket, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
    perror("udpServer bind");
  }
}

udpSender udpServer::addSender(const std::string &ip, int port) {
  return udpSender(ip.c_str(), port, _socket);
}

void udpServer::addReceiver(const std::string &ip, int port, void *buffPtr,
                            size_t len, std::function<void()> cb) {
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = inet_addr(ip.c_str());

  std::lock_guard<std::mutex> lk(_mtx);
  selector.emplace_back(buffPtr, len, addr, std::move(cb));
}

void udpServer::recvMessage() {
  uint8_t _buffer[1024] = {0};
  struct sockaddr_in src{};
  socklen_t slen = sizeof(src);

  ssize_t n =
      recvfrom(_socket, _buffer, sizeof(_buffer), 0, (struct sockaddr *)&src,
               &slen);
  if (n <= 0) return;

  std::lock_guard<std::mutex> lk(_mtx);
  for (auto &sel : selector) {
    if (src.sin_port == sel.port && sel.buffPtr != nullptr && sel.len > 0) {
      size_t copy_len = sel.len < (size_t)n ? sel.len : (size_t)n;
      memcpy(sel.buffPtr, _buffer, copy_len);
      if (sel.callback) sel.callback();
      break;
    }
  }
}

void udpServer::receiverLoop() {
  while (_running.load()) {
    recvMessage();
  }
}

void udpServer::receiverStart() {
  if (_running.load() || _socket < 0) return;
  _running.store(true);
  _rx_thread = std::thread(&udpServer::receiverLoop, this);
}

void udpServer::receiverStop() {
  _running.store(false);
  if (_socket >= 0) {
    shutdown(_socket, SHUT_RDWR);
    close(_socket);
    _socket = -1;
  }
  if (_rx_thread.joinable()) _rx_thread.join();
}
