#ifndef HOROVOD_COMMON_NET_H
#define HOROVOD_COMMON_NET_H

#include <string>
#include <iostream>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include <memory>
#include <unordered_map>

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

/*
 * Net socket class based on Linux TCP socket
 */
class Socket {
 public:
	explicit Socket (int fd): fd_(fd) {};
	Socket (const string& ip, int port): ip_(ip), port_(port) {};
	Socket (const string& ip, int port, int fd): ip_(ip), port_(port), fd_(fd) {};
	~Socket () {
	  if (fd_ > 0) {
	    close(fd_);
	  }
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
  int Init(int num_ranks);

  int Bcast(void *buffer, int size, int root = 0);

  // recvbuf should have allocated size >= size * num_ranks_
  int Gather(const void *sendbuf, int sendsize, void *recvbuf, int root = 0);

  // recvbuf should have allocated size >= Sum(recvsize)
  int Gatherv(const void *sendbuf, int sendsize,
              void *recvbuf, const int *recvsize, const int *displs, int root = 0);

  int rank() const {
    return rank_;
  }

  int num_ranks() const {
    return num_ranks_;
  }

 private:
  int rank_ = 0;
  int num_ranks_ = 0;
  bool is_master_ = false;
  string master_ip_;
  int master_port_ = 0;

  // rank to client map
  std::unordered_map<int, std::unique_ptr<ClientSocket>> clients_;
  std::unique_ptr<ServerSocket> master_;
};

} // namespace common
} // namespace horovod

#endif /* HOROVOD_COMMON_NET_H */
