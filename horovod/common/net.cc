#include <sys/types.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netdb.h>
#include <errno.h>
#include <arpa/inet.h>
#include <string.h>
#include <cstdlib>
#include <cassert>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <thread>
#include "net.h"

namespace horovod {
namespace common {

int ServerSocket::Listen() {
  int rv;
  char cport[6]; // max 65535
  addrinfo hints, *servinfo, *p;

  snprintf(cport, 6, "%d", port_);
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE; // No effect if bindaddr != nullptr

  if ((rv = getaddrinfo(ip_.size() == 0 ? nullptr : ip_.c_str(), cport, &hints, &servinfo)) != 0) {
    LOG(WARNING) << "getaddrinfo failed: " << gai_strerror(rv);
    return ST_ERROR;
  }

  for (p = servinfo; p != nullptr; p = p->ai_next) {
    if ((fd_ = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1)
      continue;

    if (bind(fd_, p->ai_addr, p->ai_addrlen) == -1) {
      LOG(FATAL) << "bind: " << strerror(errno);
      close (fd_);
      fd_ = -1;
      freeaddrinfo(servinfo);
      return ST_ERROR;
    }

    if (listen(fd_, backlog_) == -1) {
      LOG(FATAL) << "listen: " << strerror(errno);
      close (fd_);
      fd_ = -1;
      freeaddrinfo(servinfo);
      return ST_ERROR;
    }
    freeaddrinfo(servinfo);
    return ST_SUCCESS;
  }
  if (p == nullptr) {
    LOG(FATAL) << "unable to bind socket";
  }

  fd_ = -1;
  freeaddrinfo(servinfo);
  return ST_ERROR;
}

ClientSocket* ServerSocket::Accept() {
  sockaddr_storage sa;
  socklen_t salen = sizeof(sa);

  int max_ip_len = 46;
  char cip[max_ip_len];
  int cfd, cport;

  while (true) {
    cfd = accept(fd_, (sockaddr*) &sa, &salen);
    if (cfd == -1) {
      if (errno == EINTR)
        continue;
      else {
        LOG(WARNING) << "accept: " << strerror(errno);
        return nullptr;
      }
    }
    break;
  }

  if (sa.ss_family == AF_INET) {
    sockaddr_in* s = (sockaddr_in*) &sa;
    inet_ntop(AF_INET, (void*) &(s->sin_addr), cip, max_ip_len);
    cport = ntohs(s->sin_port);
  } else {
    LOG(WARNING) << "not supported IPV6";
    return nullptr;
  }

  LOG(DEBUG) << "accept " << cip << ":" << cport << " (socket = " << cfd << ")";
  return new ClientSocket (string(cip), cport, cfd);
}

int ClientSocket::Connect(bool blocking) {
  int rv;
  char portstr[6]; // strlen("65535") + 1;
  addrinfo hints, *servinfo, *bservinfo, *p, *b;

  snprintf(portstr, sizeof(portstr), "%d", port_);
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  if ((rv = getaddrinfo(ip_.c_str(), portstr, &hints, &servinfo)) != 0) {
    LOG(FATAL) << gai_strerror(rv) << ", " << strerror(errno) << "(errno = " << errno << ")";
    return ST_ERROR;
  }
  do {
    for (p = servinfo; p != NULL; p = p->ai_next) {
      /* Try to create the socket and to connect it.
       * If we fail in the socket() call, or on connect(), we retry with
       * the next entry in servinfo. */
      if ((fd_ = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1)
        continue;

      if (connect(fd_, p->ai_addr, p->ai_addrlen) == -1) {
        /* If the socket is non-blocking, it is ok for connect() to
         * return an ST_INPROCESS error here. */
        if (errno == EINPROGRESS) {
          freeaddrinfo(servinfo);
          return ST_INPROCESS;
        }
        close(fd_);
        fd_ = -1;
        LOG(TRACE) << "failed to connect" << (blocking ? ", will re-try" : "");
        continue;
      }

      /* If we ended an iteration of the for loop without errors, we
       * have a connected socket. Let's return to the caller. */
      freeaddrinfo(servinfo);
      return ST_SUCCESS;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (blocking);

  if (p == nullptr)
    LOG(WARNING) << "creating socket: " << strerror(errno);

  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
  }
  return ST_ERROR;
}

int ClientSocket::Send(const void* buf, int size) {
  int nwritten, totlen = 0;
  const char* p = static_cast<const char*>(buf);
  while (totlen != size) {
    // TODO(hzhang): it may signal SIGPIPE if remote close the connection
    nwritten = write(fd_, p, size - totlen);
    if (nwritten == 0)
      return totlen;
    if (nwritten == -1) {
      LOG(ERROR) << "Sent " << totlen << ", expected " << size;
      return -1;
    }
    totlen += nwritten;
    p += nwritten;
  }
  return totlen;
}

int ClientSocket::Recv(void* buf, int size) {
  int nread, totlen = 0;
  char* p = static_cast<char*>(buf);
  while (totlen != size) {
    nread = read(fd_, p, size - totlen);
    if (nread == 0) {
      LOG(INFO) << "socket " << fd_ << " has been closed";
      break;
    }
    if (nread == -1) {
      // TODO(hzhang): EAGAIN handle
      LOG(INFO) << "read socket " << fd_ << " failed: " << strerror(errno) << " (errno = " << errno << ")";
      break;
    }

    totlen += nread;
    p += nread;
  }
  return totlen;
}

string ClientSocket::Recv(int size) {
  char buf[size];
  int nread = Recv(buf, size);
  return nread > 0 ? string(buf, nread) : string();
}

int ClientSocket::Recv(stringstream& ss, int size) {
  char buf[size];
  int nread = Recv(buf, size);
  ss.write(buf, nread);
  return nread;
}

int SocketCommunicator::Bcast(void *buffer, int size, int root) {
  if (root == rank_) {  // master
    for (auto &it : clients_) {
      auto s = it.second->Send(buffer, size);
      if (s != size) {
        LOG(ERROR) << "Bcast failed: sent " << s << " bytes data, expected " << size << " bytes data";
        return -1;
      }
    }
  } else {  // worker
    assert(clients_.size() == 1);
    auto ret = clients_[0]->Recv(buffer, size);
    if (ret != size) {
      LOG(ERROR) << "Bcast failed: received " << ret << " bytes data, expected " << size << " bytes data";
      return -1;
    }
  }
  return 0;
}

int SocketCommunicator::Barrier(int root) {
  if (root == rank_) {  // master
    for (auto &it : clients_) {
      auto s = it.second->Send("b");
      if (s != 1) {
        LOG(ERROR) << "[Rank " << rank_ << "] Barrier failed";
        return -1;
      }
    }
  } else {  // worker
    assert(clients_.size() == 1);
    char buf;
    auto ret = clients_[0]->Recv(&buf, 1);
    if (ret != 1) {
      LOG(ERROR) << "[Rank " << rank_ << "] Barrier failed";
      return -1;
    }
  }
  return 0;
}

int SocketCommunicator::Gather(const void *sendbuf, int sendsize, void *recvbuf, int root) {
  // LOG(DEBUG, rank_) << "Gather " << sendsize << " from " << num_ranks_ << " members";
  const char *sb = static_cast<const char*>(sendbuf);
  char *rb = static_cast<char*>(recvbuf);
  if (root == rank_) {  // master
    for (int i = 1; i < num_ranks_; i++) {
      auto s = clients_.at(i)->Recv(rb + i * sendsize, sendsize);
      if (s != sendsize) {
        LOG(ERROR) << "Gather failed: received " << s
            << " bytes data, expected " << sendsize << " bytes data";
        return -1;
      }
    }
  } else {
    auto ret = clients_.at(0)->Send(sendbuf, sendsize);
    if (ret != sendsize) {
      LOG(ERROR) << "Gather failed: sent " << ret
          << " bytes data, expected " << sendsize << " bytes data";
      return -1;
    }
  }
  return 0;
}

int SocketCommunicator::Gatherv(const void *sendbuf, int sendsize,
                                void *recvbuf, const int *recvsize, const int *displs, int root) {
  const char *sb = static_cast<const char*>(sendbuf);
  char *rb = static_cast<char*>(recvbuf);
  if (root == rank_) {  // master
    assert(recvsize != nullptr);
    assert(recvbuf != nullptr);
    assert(displs != nullptr);

    auto ret = recvsize[0];
    for (int i = 1; i < num_ranks_; i++) {
      // LOG(DEBUG, rank_) << "Gatherv " << recvsize[i] << " from " << i << " to offset " << displs[i];
      auto s = clients_.at(i)->Recv(rb + displs[i], recvsize[i]);
      if (s != recvsize[i]) {
        LOG(ERROR) << "Gather failed: received " << s
            << " bytes data, expected " << recvsize[i] << " bytes data";
        return -1;
      }
    }
  } else {
    // LOG(DEBUG, rank_) << "Gatherv " << sendsize << " from " << num_ranks_ << " members";
    assert(sendbuf != nullptr);

    auto ret = clients_.at(0)->Send(sendbuf, sendsize);
    if (ret != sendsize) {
      LOG(ERROR) << "Gather failed: sent " << ret
          << " bytes data, expected " << sendsize << " bytes data";
      return -1;
    }
  }
  return 0;
}

int SocketCommunicator::Init(int num_ranks, int rank) {
  if (rank == -1) {
    if (const char* env_p = std::getenv("HOROVOD_RANK")) {
      rank_ = std::stoi(env_p);
    } else {
      LOG(WARNING) << "HOROVOD_RANK is not configured";
    }
  } else {
    rank_ = rank;
  }
  num_ranks_ = num_ranks;
  LOG(INFO) << "HOROVOD_NUM_RANKS = " << num_ranks_ << ", HOROVOD_RANK = " << rank_;

  if (rank_ == 0) {
    is_master_ = true;
  } else {
    is_master_ = false;
  }

  // FIXME(hzhang):
  // 1. for master, put the master ip:port to the central controller
  // 2. for worker, get the master ip:port from the central controller
  if (const char* env_p = std::getenv("HOROVOD_MASTER")) {
    string hp(env_p);
    std::vector<string> tokens;
    boost::split(tokens, hp, boost::is_any_of(":"));
    if (tokens.size() < 2) {
      LOG(ERROR) << "HOROVOD_MASTER format error: " << hp;
      return -1;
    }
    LOG(INFO) << "HOROVOD_MASTER is " << tokens[0] << ":" << tokens[1];
    master_ip_ = tokens[0];
    master_port_ = std::stoi(tokens[1]);
  } else {
    LOG(ERROR) << "HOROVOD_MASTER is not configured";
    return -1;
  }

  // TODO(hzhang): if is_master is true, check its ip is the same as master_uri host
  // if master, create master socket and listen
  if (is_master_) {
    LOG(INFO) << "Create master on " << master_ip_ << ":" << std::to_string(master_port_);
    master_.reset(new ServerSocket(master_port_));
    master_->Listen();

    // TODO(hzhang): rank to client map
    // establish num_ranks_ - 1 connections with all workers
    for (int i = 1; i < num_ranks_; i++) {
      auto client = std::unique_ptr<ClientSocket>(master_->Accept());
      int rank = -1;
      // receive the worker's rank
      client->Recv(&rank, sizeof(int));
      LOG(INFO) << "Connection from worker " << rank
          << " (" << client->ip() << ":" << client->port() << ")";
      clients_.emplace(rank, std::move(client));
    }
  } else {  // if worker, connect with master
    clients_.clear();
    clients_.emplace(0, std::unique_ptr<ClientSocket>(new ClientSocket(master_ip_, master_port_)));
    clients_.at(0)->Connect();
    // send its own rank to master
    clients_.at(0)->Send(&rank_, sizeof(int));
  }

  return 0;
}

SocketCommunicator::~SocketCommunicator() {
  if (rank_ == 0) {
    // FIXME(hzhang): find a better way to let client close first
    sleep(1);
    clients_.clear();
    master_.reset();
  } else {
    clients_.clear();
  }
}

} // namespace common
} // namespace horovod

