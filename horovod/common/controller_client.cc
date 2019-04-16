#include "controller_client.h"
#include "logging.h"

using namespace grpcservice;
using namespace grpc;

namespace horovod {
namespace common {

std::string ControllerClient::GetMasterURI() {
  AutobotJobNode request;
  request.set_name(job_name_);
  request.set_namespace_(ns_);

  URIReply reply;
  ClientContext context;
  Status status = stub_->GetMasterURI(&context, request, &reply);

  if (status.ok()) {
    return reply.uri();
  } else {
    LOG(ERROR) << "GetMasterURI failed: " << status.error_message() << " (" << status.error_code()
        << ")";
    return "";
  }
}

ErrorCode ControllerClient::SetMasterURI(const std::string &uri) {
  AutobotJobNode request;
  request.set_name(job_name_);
  request.set_namespace_(ns_);
  request.set_uri(uri);

  ErrorCodeReply reply;
  ClientContext context;
  Status status = stub_->SetMasterURI(&context, request, &reply);

  if (status.ok()) {
    return reply.errorcode();
  } else {
    LOG(ERROR) << "SetMasterURI failed: " << status.error_message() << " (" << status.error_code()
        << ")";
    return OTHER;
  }
}

int ControllerClient::GetNumOfRanks() {
  AutobotJobNode request;
  request.set_name(job_name_);
  request.set_namespace_(ns_);

  NumOfRankReply reply;
  ClientContext context;
  Status status = stub_->GetNumOfRanks(&context, request, &reply);

  if (status.ok()) {
    return reply.numofranks();
  } else {
    LOG(ERROR) << "GetNumOfRanks failed: " << status.error_message() << " (" << status.error_code()
        << ")";
    return 0;
  }
}

}
}
