#ifndef HOROVOD_COMMON_NET_H
#define HOROVOD_COMMON_NET_H

#include <string>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include "logging.h"

namespace horovod {
namespace common {

using std::string;
using std::stringstream;

#define TCP_BACKLOG 511

#define ST_SUCCESS 0
#define ST_ERROR -1
#define ST_CLOSED 1
#define ST_INPROCESS 2
#define DEFAULT_RECV_SIZE 10240

#if HOROVOD_USE_TIMER
class Timer {
 public:
  Timer(const std::string &name) : name_(name) {
    start_ = std::chrono::high_resolution_clock::now();
  }

  ~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
    if (microseconds > 1000) LOG(INFO) << name_ << ": " << microseconds;
  }

 private:
  std::string name_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};
#else
class Timer {
 public:
  Timer(const std::string &name) {}
};
#endif

/*
 * Net socket class based on Linux TCP socket
 */
class Socket {
 public:
  explicit Socket (int fd): fd_(fd) {};
  Socket (const string& ip, int port): ip_(ip), port_(port) {};
  Socket (const string& ip, int port, int fd): ip_(ip), port_(port), fd_(fd) {};
  ~Socket () {
    Close();
  }

  // get the file descriptor of the socket
  int GetFD() const noexcept {
    return fd_;
  }

  string ip() const noexcept {
    return ip_;
  }

  int port() const noexcept {
    return port_;
  }

  void Close() {
    if (fd_ > 0) {
      LOG(DEBUG) << "close " << ip_ << ":" << port_;
      close(fd_);
    }
    fd_ = -1;
  }

 protected:
  string ip_;
  int port_ = 0;
  int fd_ = -1;
};

class ClientSocket: public Socket {
 public:
  ClientSocket(const string& ip, int port): Socket(ip, port) {}
  ClientSocket(const string& ip, int port, int fd):
    Socket(ip, port, fd) {};

  // connect to the server
  int Connect(bool blocking = true);

  // send buf[0, size] over the socket
  int Send(const void* buf, int size);

  // send buf[0, size] over the socket
  int Send(const string& buf) {
    return Send(buf.data(), buf.size());
  }

  /*
   * read the data (max = size) from socket and put the data in buf
   * return: size of data received
   */
  int Recv(void* buf, int size = DEFAULT_RECV_SIZE);
  /*
   * read the data (max = size) from socket and put the data in a string and return
   * return: received data as a string
   */
  string Recv(int size = DEFAULT_RECV_SIZE);
  /*
   * read the data (max = size) from socket and write the data to stringstream
   * return: size of data received
   */
  int Recv(stringstream& ss, int size = DEFAULT_RECV_SIZE);
};

class ServerSocket: public Socket {
 public:
  explicit ServerSocket(int port, const string& bind_addr = "", int backlog = TCP_BACKLOG)
      : Socket(bind_addr, port), backlog_(backlog) {}

  /*
   * start listen to the socket
   * return:
   * if failed, return ST_ERROR
   * otherwise, return ST_SUCCESS
   */
  int Listen();

  /*
   * accept a client socket
   * return:
   * if failed, return nullptr
   * otherwise, return the newly created ClientSocket
   */
  ClientSocket* Accept();

 private:
  int backlog_;
};

class SocketCommunicator {
 public:
  using AllReduceOp = std::function<int(const void *a, const void *b, void *res, int size)>;

  ~SocketCommunicator();
  int Init(int rank, int num_ranks, const std::string &master_uri, int root = 0);

  void Destroy();

  int Bcast(void *buffer, int size, int root = 0, const std::vector<int> &ranks = std::vector<int>());

  // recvbuf should have allocated size >= size * num_ranks_
  int Gather(const void *sendbuf, int sendsize, void *recvbuf, int root = 0);

  // recvbuf should have allocated size >= Sum(recvsizes)
  // NOTE: recvsizes/displs are in term of byte size (not number of elements)
  int Gatherv(const void *sendbuf, int sendsize,
              void *recvbuf, const int *recvsizes, const int *displs, int root = 0);

  int AllGather(const void *sendbuf, int sendsize, void *recvbuf, int root = 0);

  // NOTE: recvsizes/displs are in terms of byte size (not number of elements)
  int AllGatherv(const void *sendbuf, int sendsize,
              void *recvbuf, const int *recvsizes, const int *displs, int root = 0);

  int Barrier(int root = 0);

  int AllReduce(
      const void *sendbuf, void *recvbuf, int sendsize, AllReduceOp op, int root = 0);

  int rank() const {
    return rank_;
  }

  int num_ranks() const {
    return num_ranks_;
  }

  static uint64_t GetHostHash(const string &str) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (char c : str) {
      result = ((result << 5) + result) + c;
    }
    return result;
  }

  static std::string GetHostName() {
    char hostname[1024];
    auto ret = gethostname(hostname, 1024);
    if (ret != 0) {
      LOG(ERROR) << "gethostname failed";
      return "";
    }

    for (int i = 0; i < 1024; i++) {
      if (hostname[i] == '.') {
        hostname[i] = '\0';
        return std::string(hostname);
      }
    }
    return std::string(hostname);
  }

  static std::string GetIp(const std::string &iface = "");

 private:
  int root_ = 0;
  int rank_ = 0;
  int num_ranks_ = 1;
  bool is_master_ = true;
  string master_ip_;
  int master_port_ = 0;

  // rank to client map
  std::unordered_map<int, std::unique_ptr<ClientSocket>> clients_;
  std::unique_ptr<ServerSocket> master_;

  void CheckRootConsistency(int root) {
    if (root != root_) {
      string msg = "Bcast root (" + std::to_string(root) + ") is not equal to init root ("
          + std::to_string(root_) + ")";
      LOG(ERROR) << msg;
      throw std::invalid_argument(msg);
    }
  }
};

} // namespace common
} // namespace horovod

#endif /* HOROVOD_COMMON_NET_H */
