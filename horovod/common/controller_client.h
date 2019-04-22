#ifndef HOROVOD_COMMON_CONTROLLER_CLIENT_H
#define HOROVOD_COMMON_CONTROLLER_CLIENT_H

#include <grpcpp/grpcpp.h>
#include <string>
#include "logging.h"
#include "grpcservice.grpc.pb.h"

namespace horovod {
namespace common {

class ControllerClient {
 public:
  ControllerClient(const std::string &uri)
      : uri_(uri) {
    auto channel = grpc::CreateChannel(uri, grpc::InsecureChannelCredentials());
    stub_ = grpcservice::AutobotOperator::NewStub(channel);
    LOG(INFO) << "Connect to central controller: " << uri_;
  }

  std::string GetMasterURI();
  int SetMasterURI(const std::string &uri);
  int GetNumOfRanks();

  grpcservice::ActionReply GetAction();
  int GraphReady();
  int ReadyToStop();

  void set_job_name(const std::string &job_name) {
    job_name_ = job_name;
    request_.set_name(job_name_);
  }

  void set_namespace_name(const std::string &ns_name) {
    ns_ = ns_name;
    request_.set_namespace_(ns_);
  }

  void set_rank(int rank) {
    rank_ = rank;
    request_.set_rank(rank_);
  }

  const std::string &job_name() const {
    return job_name_;
  }

  const std::string &namespace_name() const {
    return ns_;
  }

 private:
  std::string uri_;
  std::unique_ptr<grpcservice::AutobotOperator::Stub> stub_;
  std::string job_name_;  // job name
  std::string ns_;  // namespace
  int rank_ = -1;

  grpcservice::AutobotJobNode request_;
};

}  // namespace common
}  // namespace horovod

#endif  // HOROVOD_COMMON_CONTROLLER_CLIENT_H
