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

void TestSocket() {
  string cstr = "good";
  string sstr = "morning";

  if (FLAGS_server) {
    ServerSocket socket(FLAGS_port);
    socket.Listen();
    ClientSocket *csocket = socket.Accept();
    auto recv = csocket->Recv(cstr.size());
    std::cout << "recv " << recv << std::endl;

    recv = csocket->Recv(cstr.size());
    std::cout << "recv " << recv << std::endl;

    csocket->Send(sstr);

    recv = csocket->Recv(cstr.size());
  } else {
    ClientSocket socket(FLAGS_host, FLAGS_port);
    socket.Connect();
    socket.Send(cstr);
    socket.Send(cstr);

    auto recv = socket.Recv(sstr.size());
    std::cout << "recv = " << recv << std::endl;
  }
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  SocketCommunicator comm;
  comm.Init(FLAGS_num_ranks);
  string data = "good morning";
  char buf[100];
  if (comm.rank() == 0) {
    memcpy(buf, data.data(), data.size());
  }
  int ret = comm.Bcast(buf, data.size());
  assert(ret == 0);
  buf[data.size()] = 0;
  LOG(INFO) << "[rank = " << comm.rank() << "] Bcast result = " << buf;

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

  return 0;
}
