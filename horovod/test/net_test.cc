#include "gtest/gtest.h"
#include "../common/net.h"
#include <thread>
#include <vector>

using namespace horovod::common;
constexpr int kDefaultPort = 12345;

void SendRecv(const string &cstr, const string &sstr) {
  ServerSocket server_socket(kDefaultPort);
  ClientSocket *csocket = nullptr;
  std::thread server([&server_socket, &cstr, &sstr, &csocket] {
    server_socket.Listen();
    csocket = server_socket.Accept();
    auto recv = csocket->Recv(cstr.size());
    EXPECT_EQ(recv, cstr);

    recv = csocket->Recv(cstr.size());
    EXPECT_EQ(recv, cstr);
    csocket->Send(sstr);
  });

  ClientSocket socket("localhost", kDefaultPort);
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
    std::cout << "Run test for size " << size << std::endl;
    string str(size, 'a');
    SendRecv(str, str);
  }
}

TEST(NetTest, CommTest) {
  auto master_uri = SocketCommunicator::GetIp() + ":" + std::to_string(kDefaultPort);
  std::vector<int> sizes = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000};
  int num_ranks = 3;

  for (auto &size : sizes) {
    std::cout << "Run test for size " << size << std::endl;
    std::vector<std::thread> threads;
    std::vector<string> strs;
    string gstr;
    for (int i = 0; i < num_ranks; i++) {
      strs.emplace_back(string(size, i));
      gstr += strs.back();
    }
    for (int i = 0; i < num_ranks; i++) {
      threads.emplace_back(std::thread([&, rank=i]{
        SocketCommunicator comm;
        comm.Init(rank, num_ranks, master_uri);
        if (rank == 0) {
          int ret = comm.Bcast(const_cast<char*>(strs[rank].data()), size);
          EXPECT_EQ(ret, 0);

          char *buf = new char[size * num_ranks];
          memcpy(buf, strs[rank].data(), strs[rank].size());
          ret = comm.Gather(nullptr, strs[rank].size(), buf);
          EXPECT_EQ(ret, 0);
          EXPECT_EQ(string(buf, size * num_ranks), gstr);
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
          EXPECT_EQ(string(buf, size * (num_ranks - 1)), gstr.substr(strs[rank].size()));

          buf = new char[size * num_ranks];
          ret = comm.AllGather(strs[rank].data(), size, buf);
          EXPECT_EQ(ret, 0);
          EXPECT_EQ(string(buf, size * num_ranks), gstr);
          delete[] buf;
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

          buf = new char[size * num_ranks];
          ret = comm.AllGather(strs[rank].data(), size, buf);
          EXPECT_EQ(ret, 0);
          EXPECT_EQ(string(buf, size * num_ranks), gstr);
          delete[] buf;
        }
      }));
    }

    for (int i = 0; i < num_ranks; i++) {
      threads[i].join();
    }
  }
}

TEST(NetTest, SelectiveBcast) {
  auto master_uri = SocketCommunicator::GetIp() + ":" + std::to_string(kDefaultPort);
  int num_ranks = 3;
  std::vector<std::thread> threads;
  string str(100, 'a');
  int size = str.size();
  std::vector<int> to_bcast = {2};
  for (int i = 0; i < num_ranks; i++) {
    threads.emplace_back(std::thread([&, rank=i]{
      SocketCommunicator comm;
      comm.Init(rank, num_ranks, master_uri);
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

TEST(NetTest, PerfTest) {
  auto master_uri = SocketCommunicator::GetIp() + ":" + std::to_string(kDefaultPort);
  std::vector<int> sizes = {1, 10, 100, 1000, 10000, 100000, 1000000};
  int num_ranks = 3;
  int iter = 1000;
  for (auto &size : sizes) {
    std::vector<std::thread> threads;
    std::vector<string> strs;
    string gstr;
    for (int i = 0; i < num_ranks; i++) {
      strs.emplace_back(string(size, i));
      gstr += strs.back();
    }
    for (int i = 0; i < num_ranks; i++) {
      threads.emplace_back(std::thread([&, rank=i] {
        SocketCommunicator comm;
        comm.Init(rank, num_ranks, master_uri);

        auto start = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
          for (int it = 0; it < iter; it++) {
            int ret;
            char *buf = nullptr;
            ret = comm.Bcast(const_cast<char*>(strs[rank].data()), size);
            EXPECT_EQ(ret, 0);

            buf = new char[size * num_ranks];
            memcpy(buf, strs[rank].data(), strs[rank].size());
            ret = comm.Gather(nullptr, strs[rank].size(), buf);
            EXPECT_EQ(ret, 0);
            EXPECT_EQ(string(buf, size * num_ranks), gstr);
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
            EXPECT_EQ(string(buf, size * (num_ranks - 1)), gstr.substr(strs[rank].size()));
            delete[] buf;
          }
        } else {
          char *buf = new char[size];
          for (int it = 0; it < iter; it++) {
            int ret;
            ret = comm.Bcast(buf, size);
            EXPECT_EQ(ret, 0);
            EXPECT_EQ(string(buf, size), strs[0]);

            ret = comm.Gather(strs[rank].data(), strs[rank].size(), nullptr);
            EXPECT_EQ(ret, 0);

            ret = comm.Gatherv(strs[rank].data(), strs[rank].size(), nullptr, nullptr, nullptr);
            EXPECT_EQ(ret, 0);
          }
          delete[] buf;
        }
        auto end = std::chrono::high_resolution_clock::now();
        int64_t microseconds =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (rank == 0)
          std::cout << "Perf [" << rank << "] (size = " << size << "): "
              << (static_cast<long>(iter) * 1000 * 1000 / microseconds) << " iter/s" << std::endl;
      }));
    }

    for (int i = 0; i < num_ranks; i++) {
      threads[i].join();
    }
  }
}

TEST(NetTest, ReInitTest) {
  auto master_uri = SocketCommunicator::GetIp() + ":" + std::to_string(kDefaultPort);
  std::vector<int> rank_choices = {1, 2, 3, 4};
  int size = 1024;

  SocketCommunicator comms[4];
  for (auto &num_ranks: rank_choices) {
    std::cout << "Run test for num_ranks " << num_ranks << std::endl;
    std::vector<std::thread> threads;
    std::vector<string> strs;
    string gstr;
    for (int i = 0; i < num_ranks; i++) {
      strs.emplace_back(string(size, i));
      gstr += strs.back();
    }
    for (int i = 0; i < num_ranks; i++) {
      threads.emplace_back(std::thread([&, rank=i] {
        auto &comm = comms[rank];
        comm.Init(rank, num_ranks, master_uri);
        if (rank == 0) {
          int ret = comm.Bcast(const_cast<char*>(strs[rank].data()), size);
          EXPECT_EQ(ret, 0);

          char *buf = new char[size * num_ranks];
          memcpy(buf, strs[rank].data(), strs[rank].size());
          ret = comm.Gather(nullptr, strs[rank].size(), buf);
          EXPECT_EQ(ret, 0);
          EXPECT_EQ(string(buf, size * num_ranks), gstr);
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
          EXPECT_EQ(string(buf, size * (num_ranks - 1)), gstr.substr(strs[rank].size()));

          buf = new char[size * num_ranks];
          ret = comm.AllGather(strs[rank].data(), size, buf);
          EXPECT_EQ(ret, 0);
          EXPECT_EQ(string(buf, size * num_ranks), gstr);
          delete[] buf;
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

          buf = new char[size * num_ranks];
          ret = comm.AllGather(strs[rank].data(), size, buf);
          EXPECT_EQ(ret, 0);
          EXPECT_EQ(string(buf, size * num_ranks), gstr);
          delete[] buf;
        }
        comm.Destroy();
      }));
    }

    for (int i = 0; i < num_ranks; i++) {
      threads[i].join();
    }
  }
}
