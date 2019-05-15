#include <string.h>
#include <cassert>
#include "net.h"
#include "gflags/gflags.h"
#include "logging.h"

using namespace horovod::common;

DEFINE_bool(server, true, "if it is a server");
DEFINE_int32(port, 12345, "port number");
DEFINE_string(host, "localhost", "host for the server");
DEFINE_int32(num_ranks, 3, "number of ranks");
DEFINE_int32(size, 1000, "data size");
DEFINE_int32(iter, 1000, "iter");

void TestSocket(int size, int iter, const std::string &ip) {
  string cstr = string(size, 'a'); // "good";
  string sstr = string(size, 'b'); // "morning";

  bool is_server = ip.empty();

  if (is_server) {
    ServerSocket socket(12345);
    socket.Listen();
    ClientSocket *csocket = socket.Accept();

    Timer t("Server");
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++) {
      auto recv = csocket->Recv(cstr.size());
      // std::cout << "recv " << recv << std::endl;

      // recv = csocket->Recv(cstr.size());
      // std::cout << "recv " << recv << std::endl;

      csocket->Send(sstr);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Message/s: " << (iter * 1000L * 1000 / microseconds) << std::endl;
  } else {
    ClientSocket socket(ip, 12345);
    socket.Connect();
    Timer t("Client");
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++) {
      socket.Send(cstr);
      // socket.Send(cstr);

      auto recv = socket.Recv(sstr.size());
      // std::cout << "recv = " << recv << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Message/s: " << (iter * 1000L * 1000 / microseconds) << std::endl;
  }
}

int main(int argc, char *argv[]) {
  // if (argc > 3) {
  //   TestSocket(std::stoi(argv[1]), std::stoi(argv[2]), argv[3]);
  // } else {
  //   TestSocket(std::stoi(argv[1]), std::stoi(argv[2]), "");
  //   sleep(1);
  // }
  // return 0;

  google::ParseCommandLineFlags(&argc, &argv, true);
  SocketCommunicator comm;

  int rank = -1;
  if (const char* env_p = std::getenv("AUTOBOT_RANK")) {
    rank = std::stoi(env_p);
  } else {
    LOG(WARNING) << "AUTOBOT_RANK is not configured";
  }
  comm.Init(rank, FLAGS_num_ranks, FLAGS_host + ":" + std::to_string(FLAGS_port));
  string data = string(FLAGS_size, 'a');
  char *buf = new char[FLAGS_size + 1];
  if (comm.rank() == 0) {
    memcpy(buf, data.data(), data.size());
  }
  int ret = comm.Bcast(buf, data.size());
  assert(ret == 0);
  buf[data.size()] = 0;
  LOG(INFO) << "[rank = " << comm.rank() << "] Bcast result = " << buf;

  {
    Timer t("Broadcast");
    for (int i = 0; i < FLAGS_iter; i++) {
      int ret = comm.Bcast(buf, data.size());
      assert(ret == 0);
    }
  }

  int recvcounts[FLAGS_num_ranks];
  recvcounts[0] = 0;
  string to_send = "prefix_" + std::to_string(comm.rank() * 10) + "_subfix";
  int to_send_size = to_send.size();
  if (comm.rank() == 0) {
    ret = comm.Gather(nullptr, sizeof(int), recvcounts);
    assert(ret == 0);
    std::cout << "[rank = " << comm.rank() << "] Gather result: ";
    for (int i = 0; i < FLAGS_num_ranks; i++) {
      std::cout << recvcounts[i] << ", ";
    }
    std::cout << std::endl;
  } else {
    ret = comm.Gather(&to_send_size, sizeof(int), nullptr);
    assert(ret == 0);
  }

  if (comm.rank() == 0) {
    int total_size = 0;
    int displcmnts[FLAGS_num_ranks];
    for (int i = 0; i < FLAGS_num_ranks; ++i) {
      if (i == 0) {
        displcmnts[i] = 0;
      } else {
        displcmnts[i] = recvcounts[i - 1] + displcmnts[i - 1];
      }
      total_size += recvcounts[i];
    }
    char recvbuf[total_size];
    ret = comm.Gatherv(nullptr, 0, recvbuf, recvcounts, displcmnts);
    assert(ret == 0);

    std::cout << "[rank = " << comm.rank() << "] Gatherv result: ";
    for (int i = 1; i < FLAGS_num_ranks; i++) {
      auto ptr = recvbuf + displcmnts[i];
      std::cout << string(ptr, recvcounts[i]) << ", ";
    }
    std::cout << std::endl;
  } else {
    ret = comm.Gatherv(to_send.data(), to_send_size, nullptr, nullptr, nullptr);
    assert(ret == 0);
  }

  delete[] buf;
  return 0;
}
