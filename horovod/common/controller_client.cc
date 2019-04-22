#include "controller_client.h"

#include <thread>
#include "logging.h"

using namespace grpcservice;
using namespace grpc;

namespace horovod {
namespace common {

std::string ControllerClient::GetMasterURI() {
  int retries = 20;
  for (int i = 0; i < retries; i++) {
    URIReply reply;
    ClientContext context;
    Status status = stub_->GetMasterURI(&context, request_, &reply);

    if (status.ok()) {
      auto uri = reply.uri();
      if (!uri.empty()) return uri;
    } else {
      LOG(ERROR) << "GetMasterURI failed: " << status.error_message() << " (" << status.error_code()
          << ")";
      return "";
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  LOG(ERROR) << "GetMasterURI failed after " << retries;
  return "";
}

ErrorCode ControllerClient::SetMasterURI(const std::string &uri) {
  request_.set_uri(uri);

  ErrorCodeReply reply;
  ClientContext context;
  Status status = stub_->SetMasterURI(&context, request_, &reply);

  if (status.ok()) {
    return reply.errorcode();
  } else {
    LOG(ERROR) << "SetMasterURI failed: " << status.error_message() << " (" << status.error_code()
        << ")";
    return OTHER;
  }
}

int ControllerClient::GetNumOfRanks() {
  NumOfRankReply reply;
  ClientContext context;
  Status status = stub_->GetNumOfRanks(&context, request_, &reply);

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
  ClientContext context;
  Status status = stub_->GetAction(&context, request_, &reply);

  if (status.ok()) {
  } else {
    LOG(ERROR) << "GetAction failed: " << status.error_message() << " (" << status.error_code()
        << ")";
  }
  return reply;
}

void ControllerClient::GraphReady() {
  ErrorCodeReply reply;
  ClientContext context;
  Status status = stub_->GraphReady(&context, request_, &reply);

  if (status.ok()) {
    if (reply.errorcode() != SUCCESS) {
      LOG(ERROR) << "GraphReady failed: " << reply.errorcode();
    }
  } else {
    LOG(ERROR) << "GraphReady failed: " << status.error_message() << " (" << status.error_code()
        << ")";
  }
}

void ControllerClient::ReadyToStop() {
  ErrorCodeReply reply;
  ClientContext context;
  Status status = stub_->ReadyToStop(&context, request_, &reply);

  if (status.ok()) {
    if (reply.errorcode() != SUCCESS) {
      LOG(ERROR) << "ReadyToStop failed: " << reply.errorcode();
    }
  } else {
    LOG(ERROR) << "ReadyToStop failed: " << status.error_message() << " (" << status.error_code()
        << ")";
  }
}

}
}
