protoc -I third_party/autobot-operator/pkg/grpcservice --grpc_out=horovod/common --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` third_party/autobot-operator/pkg/grpcservice/grpcservice.proto
protoc -I third_party/autobot-operator/pkg/grpcservice --cpp_out=horovod/common third_party/autobot-operator/pkg/grpcservice/grpcservice.proto
