#ifndef HOROVOD_SOCKET_OPERATIONS_H
#define HOROVOD_SOCKET_OPERATIONS_H

#include "../common.h"
#include "../global_state.h"
#include "../mpi_context.h"
#include "collective_operations.h"

namespace horovod {
namespace common {

class SocketAllgather : public AllgatherOp {
public:
  SocketAllgather(SocketContext* socket_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override {
    return entries[0].device == CPU_DEVICE_ID;
  }

protected:
  SocketContext* socket_context_;
};

} // namespace common
} // namespace horovod

#endif  // HOROVOD_SOCKET_OPERATIONS_H
