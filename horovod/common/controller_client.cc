#include "controller_client.h"

#include <thread>
#include "logging.h"

using namespace grpcservice;
using namespace grpc;

namespace horovod {
namespace common {

std::string ControllerClient::GetMasterURI() {
  int retries = 0;
  while (true) {
    URIReply reply;
    auto query = [this, &reply] () {
      ClientContext context;
      Status status = stub_->GetMasterURI(&context, request_, &reply);
      return status;
    };
    auto status = QueryWithRetries(query);

    if (status.ok()) {
      auto uri = reply.uri();
      if (!uri.empty()) return uri;
    } else {
      LOG(ERROR) << "GetMasterURI failed: " << status.error_message() << " (" << status.error_code()
          << ")";
      return "";
    }

    retries++;
    LOG(INFO) << "GetMasterURI failed " << retries << " times";
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }
  return "";
}

int ControllerClient::SetMasterURI(const std::string &uri) {
  request_.set_uri(uri);

  ErrorCodeReply reply;
  auto query = [this, &reply] () {
    ClientContext context;
    Status status = stub_->SetMasterURI(&context, request_, &reply);
    return status;
  };
  auto status = QueryWithRetries(query);

  if (status.ok()) {
    return reply.errorcode();
  } else {
    LOG(ERROR) << "SetMasterURI failed: " << status.error_message() << " (" << status.error_code()
        << ")";
    return -1;
  }
}

int ControllerClient::GetNumOfRanks() {
  NumOfRankReply reply;
  auto query = [this, &reply] () {
    ClientContext context;
    Status status = stub_->GetNumOfRanks(&context, request_, &reply);
    return status;
  };
  auto status = QueryWithRetries(query);

  if (status.ok()) {
    return reply.numofranks();
  } else {
    LOG(ERROR) << "GetNumOfRanks failed: " << status.error_message() << " (" << status.error_code()
        << ")";
    return 0;
  }
}

ActionReply ControllerClient::GetAction() {
  ActionReply reply;
  auto query = [this, &reply] () {
    ClientContext context;
    Status status = stub_->GetAction(&context, request_, &reply);
    return status;
  };
  auto status = QueryWithRetries(query);

  if (!status.ok()) {
    string msg = "GetAction failed: " + status.error_message()
        + " (" + std::to_string(status.error_code()) + ")";
    LOG(ERROR) << msg;
    throw std::invalid_argument(msg);
  }
  return reply;
}

int ControllerClient::GraphReady() {
  ErrorCodeReply reply;
  auto query = [this, &reply] () {
    ClientContext context;
    Status status = stub_->GraphReady(&context, request_, &reply);
    return status;
  };
  auto status = QueryWithRetries(query);

  if (status.ok()) {
    if (reply.errorcode() != SUCCESS) {
      LOG(ERROR) << "GraphReady failed: " << reply.errorcode();
    }
    return reply.errorcode();
  } else {
    LOG(ERROR) << "GraphReady failed: " << status.error_message() << " (" << status.error_code()
        << ")";
    return -1;
  }
}

int ControllerClient::ReadyToStop() {
  ErrorCodeReply reply;
  auto query = [this, &reply] () {
    ClientContext context;
    Status status = stub_->ReadyToStop(&context, request_, &reply);
    return status;
  };

  Status status = QueryWithRetries(query);

  if (status.ok()) {
    if (reply.errorcode() != SUCCESS) {
      LOG(ERROR) << "ReadyToStop failed: " << reply.errorcode();
    }
    return reply.errorcode();
  } else {
    LOG(ERROR) << "ReadyToStop failed: " << status.error_message() << " (" << status.error_code()
        << ")";
    return -1;
  }
}

Status ControllerClient::QueryWithRetries(std::function<Status()> func) {
  int retries = 0;
  while (true) {
    auto status = func();
    auto code = status.error_code();
    if (code != grpc::UNAVAILABLE && code != grpc::CANCELLED) {
      return status;
    }
    retries++;

    LOG(WARNING) << "Connect to controller failed (" << status.error_message() << ") "
        << retries << " times. Will retry";
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }
}

}
}
