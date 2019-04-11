#include "gtest/gtest.h"
#include "../common/net.h"
#include <thread>
#include <vector>

using namespace horovod::common;

void SendRecv(const string &cstr, const string &sstr) {
  int port = 12345;
  ServerSocket server_socket(port);
  ClientSocket *csocket = nullptr;
  std::thread server([&server_socket, port, &cstr, &sstr, &csocket] {
    server_socket.Listen();
    csocket = server_socket.Accept();
    auto recv = csocket->Recv(cstr.size());
    EXPECT_EQ(recv, cstr);

    recv = csocket->Recv(cstr.size());
    EXPECT_EQ(recv, cstr);
    csocket->Send(sstr);
  });

  ClientSocket socket("localhost", port);
  socket.Connect();
  socket.Send(cstr);
  socket.Send(cstr);

  auto recv = socket.Recv(sstr.size());
  EXPECT_EQ(recv, sstr);

  socket.Close();
  csocket->Close();
  server.join();
  server_socket.Close();
}

void Broadcast(SocketCommunicator &comm, void *buf, int size) {
  comm.Bcast(buf, size);
}

TEST(NetTest, SocketTest) {
  std::vector<int> sizes = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000};
  for (auto &size : sizes) {
    string str(size, 'a');
    SendRecv(str, str);
  }
}

TEST(NetTest, CommTest) {
  std::vector<int> sizes = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000};
  int num_ranks = 3;
  for (auto &size : sizes) {

    std::vector<std::thread> threads;
    std::vector<string> strs;
    for (int i = 0; i < num_ranks; i++) {
      strs.emplace_back(string(size, i));
      threads.emplace_back(std::thread([&, rank=i]{
        SocketCommunicator comm;
        comm.Init(num_ranks, rank);
        if (rank == 0) {
          int ret = comm.Bcast(const_cast<char*>(strs[rank].data()), size);
          EXPECT_EQ(ret, 0);

          char *buf = new char[size * num_ranks];
          memcpy(buf, strs[rank].data(), strs[rank].size());
          ret = comm.Gather(nullptr, strs[rank].size(), buf);
          EXPECT_EQ(ret, 0);
          EXPECT_EQ(string(buf, size * num_ranks), strs[0] + strs[1] + strs[2]);
          delete[] buf;

          int displcmnts[num_ranks];
          int recvcounts[num_ranks];
          recvcounts[0] = 0;
          for (int j = 1; j < num_ranks; j++) {
            recvcounts[j] = strs[j].size();
          }
          displcmnts[0] = 0;
          for (int j = 1; j < num_ranks; j++) {
            displcmnts[j] = displcmnts[j - 1] +  recvcounts[j - 1];
          }
          buf = new char[size * (num_ranks - 1)];
          ret = comm.Gatherv(nullptr, 0, buf, recvcounts, displcmnts);
          EXPECT_EQ(ret, 0);
          EXPECT_EQ(string(buf, size * (num_ranks - 1)), strs[1] + strs[2]);
        } else {
          char *buf = new char[size];
          int ret = comm.Bcast(buf, size);
          EXPECT_EQ(ret, 0);
          EXPECT_EQ(string(buf, size), strs[0]);
          delete[] buf;

          ret = comm.Gather(strs[rank].data(), strs[rank].size(), nullptr);
          EXPECT_EQ(ret, 0);

          ret = comm.Gatherv(strs[rank].data(), strs[rank].size(), nullptr, nullptr, nullptr);
          EXPECT_EQ(ret, 0);
        }
      }));
    }

    for (int i = 0; i < num_ranks; i++) {
      threads[i].join();
    }
  }
}

TEST(NetTest, SelectiveBcast) {
  int num_ranks = 3;
  std::vector<std::thread> threads;
  string str(100, 'a');
  int size = str.size();
  std::vector<int> to_bcast = {2};
  for (int i = 0; i < num_ranks; i++) {
    threads.emplace_back(std::thread([&, rank=i]{
      SocketCommunicator comm;
      comm.Init(num_ranks, rank);
      if (rank == 0) {
        int ret = comm.Bcast(const_cast<char*>(str.data()), str.size(), 0, to_bcast);
        EXPECT_EQ(ret, 0);
      } else if (rank == 2) {
        char *buf = new char[size];
        int ret = comm.Bcast(buf, size);
        EXPECT_EQ(ret, 0);
        EXPECT_EQ(string(buf, size), str);
        delete[] buf;
      }
    }));
  }

  for (int i = 0; i < num_ranks; i++) {
    threads[i].join();
  }
}
