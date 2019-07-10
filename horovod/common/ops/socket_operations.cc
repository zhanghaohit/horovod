#include "socket_operations.h"
#include "../logging.h"
#include "../half.h"

namespace horovod {
namespace common {

SocketAllgather::SocketAllgather(SocketContext* socket_context, HorovodGlobalState* global_state)
    : AllgatherOp(global_state), socket_context_(socket_context) {}

Status SocketAllgather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  LOG(DEBUG) << "SocketAllgather";
  auto& timeline = global_state_->timeline;

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t* [entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t* [entries.size()];

  auto* recvcounts = new int[global_state_->size]();
  auto* displcmnts = new int[global_state_->size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_state_->size]();
    entry_component_offsets[ec] = new int64_t[global_state_->size]();
  }

  auto& first_entry = entries[0];

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (!status.ok()) {
    return status;
  }
  timeline.ActivityEndAll(entries);

  SetDisplacements(recvcounts, displcmnts);
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts, entry_component_offsets);

  int element_size = GetSizeof(first_entry.tensor);
  const void* sendbuf = nullptr;
  void* buffer_data;
  int64_t total_num_elements = NumElements(entries);

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);
    timeline.ActivityEndAll(entries);
  } else {
    sendbuf = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
  }

  global_state_->timeline.ActivityStartAll(entries, SOCKET_ALLGATHER);
  // transform element-wise count to byte-wise counts
  auto* recvsizes = new int[global_state_->size]();
  auto* displcmntsizes = new int[global_state_->size]();
  for (int i = 0; i < global_state_->size; i++) {
    recvsizes[i] = recvcounts[i] * element_size;
    displcmntsizes[i] = displcmnts[i] * element_size;
  }

  int op = socket_context_->comm.AllGatherv(
      sendbuf != nullptr ? sendbuf : (uint8_t*)buffer_data + displcmntsizes[global_state_->rank],
      (int) total_num_elements * element_size,
      buffer_data, recvsizes, displcmntsizes);

  if (op != 0) {
    throw std::logic_error("Socket::Allgatherv failed.");
  }
  global_state_->timeline.ActivityEndAll(entries);

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                          buffer_data, element_size, entries);
    timeline.ActivityEndAll(entries);
  }

  delete[] recvcounts;
  delete[] displcmnts;
  delete[] recvsizes;
  delete[] displcmntsizes;

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    delete[] entry_component_sizes[ec];
    delete[] entry_component_offsets[ec];
  }
  delete[] entry_component_sizes;
  delete[] entry_component_offsets;

  return Status::OK();
}

SocketBroadcast::SocketBroadcast(SocketContext* socket_context, HorovodGlobalState* global_state) :
    socket_context_(socket_context), BroadcastOp(global_state) {}

Status SocketBroadcast::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  LOG(DEBUG) << "Using socket Bcast";

  assert(entries.size() == 1);
  auto e = entries[0];

  // On root rank, NCCL_Bcast sends data, on other ranks it receives data.
  void* data_ptr;
  if (global_state_->rank == e.root_rank) {
    data_ptr = (void*) e.tensor->data();
  } else {
    data_ptr = (void*) e.output->data();
  }

  global_state_->timeline.ActivityStartAll(entries, SOCKET_BCAST);
  auto ret = socket_context_->comm.Bcast(
      data_ptr, e.tensor->shape().num_elements() * GetSizeof(e.tensor), e.root_rank, e.ranks);
  if (ret != 0) {
    throw std::logic_error("Socket_Broadcast failed.");
  }
  global_state_->timeline.ActivityEndAll(entries);
  return Status::OK();
}

SocketAllreduce::SocketAllreduce(SocketContext* socket_context, HorovodGlobalState* global_state)
    : AllreduceOp(global_state), socket_context_(socket_context) {}

void SocketAllreduce::MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                             const TensorTableEntry& e, void* buffer_data_at_offset) {
  std::memcpy(buffer_data_at_offset, e.tensor->data(), (size_t) e.tensor->size());
}

void SocketAllreduce::MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                              const void* buffer_data_at_offset, TensorTableEntry& e) {
  std::memcpy((void*) e.output->data(), buffer_data_at_offset, (size_t) e.tensor->size());
}

Status SocketAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  LOG(DEBUG) << "SocketAllReduce";
  auto& first_entry = entries[0];

  void* buffer_data;
  size_t buffer_len;
  int64_t num_elements = NumElements(entries);

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, SOCKET_ALLREDUCE);
  const void* sendbuf = entries.size() > 1 || first_entry.tensor->data() == first_entry.output->data()
                        ? buffer_data : first_entry.tensor->data();

  auto op = [dtype=first_entry.tensor->dtype()](const void *a, const void *b, void *res, int size) -> int {
    assert(size % GetSizeof(dtype) == 0);
    if (dtype == HOROVOD_FLOAT16) {
      memcpy(res, a, size);
      int num = size / GetSizeof(dtype);
#ifdef DYNAMIC_SCHEDULE
      float16_sum(const_cast<void*>(b), res, &num);  // b will not be modified
#else
      float16_sum(const_cast<void*>(b), res, &num, nullptr);  // b will not be modified
#endif
    } else {
      for (int i = 0; i < size / GetSizeof(dtype); i++) {
        switch (dtype) {
          case HOROVOD_INT32:
            {
              auto ia = static_cast<const int*>(a);
              auto ib = static_cast<const int*>(b);
              auto ires = static_cast<int*>(res);
              *(ires + i) = *(ia + i) + *(ib + i);
              break;
            }
          case HOROVOD_INT64:
            {
              auto ia = static_cast<const int64_t*>(a);
              auto ib = static_cast<const int64_t*>(b);
              auto ires = static_cast<int64_t*>(res);
              *(ires + i) = *(ia + i) + *(ib + i);
              break;
            }
          case HOROVOD_FLOAT32:
            {
              auto ia = static_cast<const float*>(a);
              auto ib = static_cast<const float*>(b);
              auto ires = static_cast<float*>(res);
              *(ires + i) = *(ia + i) + *(ib + i);
              break;
            }
          case HOROVOD_FLOAT64:
            {
              auto ia = static_cast<const double*>(a);
              auto ib = static_cast<const double*>(b);
              auto ires = static_cast<double*>(res);
              *(ires + i) = *(ia + i) + *(ib + i);
              break;
            }
          default:
            throw std::logic_error("Type " + DataType_Name(dtype) + " is not supported.");
        }
      }
    }
    return 0;
  };
  int ret = socket_context_->comm.AllReduce(
      sendbuf, buffer_data, (int) num_elements * GetSizeof(first_entry.tensor), op);
  if (ret != 0) {
    throw std::logic_error("Socket_Allreduce failed.");
  }
  timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}

} // namespace common
} // namespace horovod
