#include <gtest/gtest.h>
#include "../common/controller_client.h"

using namespace horovod::common;
using namespace grpcservice;
using std::string;

TEST(ControllerClientTest, MasterURI) {
  string job_name = "dynamic-autobotjob";
  string ns_name = "video-structure";
  ControllerClient client("10.80.22.18:9001");
  client.set_job_name(job_name);
  client.set_namespace_name(ns_name);
  string master_uri = "localhost:12345";
  auto ret = client.SetMasterURI(master_uri);
  EXPECT_EQ(ret, SUCCESS);

  auto uri = client.GetMasterURI();
  EXPECT_EQ(uri, master_uri);

  auto num_of_ranks = client.GetNumOfRanks();
  std::cout << "num_of_ranks = " << num_of_ranks << std::endl;
  // EXPECT_EQ(num_of_ranks, 2);

  auto reply = client.GetAction();
  std::cout << reply.action() << std::endl;

  client.GraphReady();
  client.GetAction();
  client.ReadyToStop();
}
