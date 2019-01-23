/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef MXNET_NGRAPH_NGRAPH_NNVM_UTILS_H_
#define MXNET_NGRAPH_NGRAPH_NNVM_UTILS_H_

#include <mxnet/ndarray.h>
#include <mxnet/op_attr_types.h>

#include <algorithm>
#include <functional>
#include <vector>

#include "ngraph_sgcompiler_utils.h"
#include "ngraph_utils.h"

namespace ngraph_bridge {

using TensorVector = std::vector<std::shared_ptr<ngraph::runtime::Tensor>>;
using ngraph::runtime::Tensor;

// Simple utility for getting the total number of bytes in a
// buffer, either from an mxnet tensor or an ngraph tensor
template <typename T>
inline size_t get_buffer_size(const T& shape, size_t nbytes) {
  return std::accumulate(shape.begin(), shape.end(), nbytes,
                         std::multiplies<size_t>());
}

// This function creates an ngraph Tensor from the shape and type
// of an input mxnet TBlob. It optionally copies the data
// from the TBlob to the ngraph tensor.
inline std::shared_ptr<Tensor> NDArray_to_Tensor(
    const mxnet::NDArray& input,
    std::shared_ptr<ngraph::runtime::Backend> backend, bool copy) {
  auto shape = TShape_to_NShape(input.shape());
  const auto& element_type = getType(input.dtype());

  // TODO(mbrookhart): I don't think Ashok's memory sharing PR has a
  // create_tensor implementation? this will probably conflict
  auto TV = backend->create_tensor(element_type, shape);

  if (copy) {
    check(input.storage_handle().dptr != nullptr);
    auto buffer_size = get_buffer_size(shape, element_type.size());
    TV->write(input.storage_handle().dptr, 0, buffer_size);
  }

  return TV;
}

// Main utility funciton for creating NNVM ops
// This function takes a vector of TBlobs and creates a vector of
// equialently shaped and typed ngraph tensors, optionally
// copied the data from the TBlobs to ngraph
inline TensorVector make_ngraph_placeholders(
    const std::vector<mxnet::NDArray>& inputs,
    std::shared_ptr<ngraph::runtime::Backend> backend, bool copy_data) {
  TensorVector out;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(out),
                 [copy_data, backend](const mxnet::NDArray& input) {
                   return NDArray_to_Tensor(input, backend, copy_data);
                 });
  return out;
}

inline void init_tensors(std::shared_ptr<Graph>& graph,
                         const std::vector<mxnet::NDArray>& ndarrays) {
  graph->tensors_.assign(ndarrays.size(), nullptr);
  graph->ndarray_vers_.assign(ndarrays.size(),
                              std::numeric_limits<size_t>::max());
  auto backend = graph->get_backend();
  void* dptr = ndarrays[index].storage_handle().dptr;
  check(backend != nullptr);
  check(dptr != nullptr);
  ngraph::Shape shape{};
  if (!is_scalar) {
    shape = TShape_to_NShape(ndarrays[index].shape());
  }
  if (is_boolean) {
    graph->tensors_[index] =
        backend->create_tensor(ngraph::element::boolean, shape, dptr);

  } else {
    graph->tensors_[index] =
        backend->create_tensor(getType(ndarrays[index].dtype()), shape, dptr);
  }
  graph->ndarray_vers_[index] = ndarrays[index].version();
}
inline std::shared_ptr<ngraph::runtime::Tensor> get_tensor(
    std::shared_ptr<Graph>& graph, const std::vector<mxnet::NDArray>& ndarrays,
    size_t index, bool is_boolean, bool is_scalar) {
  if (ndarrays.size() < 1) return nullptr;
  if (graph->tensors_.size() != ndarrays.size()) {
    std::cout << "ndarrays size " << ndarrays.size() << ":"
              << graph->tensors_.size() << "\n";
    // reset tensors state
    graph->tensors_.assign(ndarrays.size(), nullptr);
    graph->ndarray_vers_.assign(ndarrays.size(),
                                std::numeric_limits<size_t>::max());
  }
  if (graph->tensors_[index] == nullptr ||
      graph->ndarray_vers_[index] != ndarrays[index].version() ||
      graph->tensors_[index]->get_shape() !=
          TShape_to_NShape(ndarrays[index].shape())) {
    std::cout << "creating ndarrays tensor  " << index << ":"
              << graph->tensors_[index] << ":" << graph->ndarray_vers_[index]
              << ":" << ndarrays[index].version() << "\n";
    auto backend = graph->get_backend();
    void* dptr = ndarrays[index].storage_handle().dptr;
    check(backend != nullptr);
    check(dptr != nullptr);
    ngraph::Shape shape{};
    if (!is_scalar) {
      shape = TShape_to_NShape(ndarrays[index].shape());
    }
    if (is_boolean) {
      graph->tensors_[index] =
          backend->create_tensor(ngraph::element::boolean, shape, dptr);

    } else {
      graph->tensors_[index] =
          backend->create_tensor(getType(ndarrays[index].dtype()), shape, dptr);
    }
  }
  graph->ndarray_vers_[index] = ndarrays[index].version();
  return graph->tensors_[index];
}

// creates and returns vector of Tensors for corresponding NDArrays
// reuses NDArray memory for each Tensor if req is not kAddTo
inline TensorVector get_tensors(
    const std::vector<mxnet::NDArray>& ndarrays, std::shared_ptr<Graph>& graph,
    std::vector<bool> is_boolean, std::vector<bool> is_scalar,
    const std::vector<mxnet::OpReqType>* req = nullptr,
    const bool mem_reuse = true) {
  auto backend = graph->get_backend();
  TensorVector out;
  for (size_t i = 0; i < ndarrays.size(); ++i) {
    if (!mem_reuse || ((req != nullptr) && ((*req)[i] == mxnet::kAddTo))) {
      out.push_back(NDArray_to_Tensor(ndarrays[i], backend, (req == nullptr)));
    } else {
      /* out.push_back(const_cast<mxnet::NDArray&>(ndarrays[i]) */
      /*                   .create_tensor(is_boolean[i], is_scalar[i])); */
      out.push_back(
          get_tensor(graph, ndarrays, i, is_boolean[i], is_scalar[i]));
    }
  }
  return out;
}
template <class T>
inline void result_plus_NDArray(void* mxnet_ptr, void* ngraph_ptr,
                                size_t buffer_size) {
  T* mxnet_ptr_tptr = static_cast<T*>(mxnet_ptr);
  T* ngraph_ptr_tptr = static_cast<T*>(ngraph_ptr);
  for (size_t i = 0; i < (buffer_size / sizeof(T)); ++i) {
    *(mxnet_ptr_tptr + i) += *(ngraph_ptr_tptr + i);
  }
}

// Utility function that copies all results from an
// ngraph computation into the output NDArrays in mxnet
inline void result_to_NDArray(
    const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& results,
    const std::vector<mxnet::OpReqType>& req,
    const std::vector<mxnet::NDArray>& outputs, bool force_read = false) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (req[i] == mxnet::kNullOp) continue;

    const auto& element_type = getType(outputs[i].dtype());
    auto buffer_size = get_buffer_size(outputs[i].shape(), element_type.size());

    void* mxnet_ndarray = outputs[i].storage_handle().dptr;
    check(mxnet_ndarray != nullptr);
    if (req[i] == mxnet::kAddTo) {
      void* ngraph_tv = malloc(buffer_size);
      check(ngraph_tv != nullptr);
      results[i]->read(ngraph_tv, 0, buffer_size);

      if (element_type == ngraph::element::f32)
        result_plus_NDArray<float>(mxnet_ndarray, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::f64)
        result_plus_NDArray<double>(mxnet_ndarray, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::u8)
        result_plus_NDArray<uint8_t>(mxnet_ndarray, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::i8)
        result_plus_NDArray<int8_t>(mxnet_ndarray, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::i32)
        result_plus_NDArray<int32_t>(mxnet_ndarray, ngraph_tv, buffer_size);
      else if (element_type == ngraph::element::i64)
        result_plus_NDArray<int64_t>(mxnet_ndarray, ngraph_tv, buffer_size);

      free(ngraph_tv);
    } else {
      // TODO(adstraw): Add support for kWriteInplace
      if (force_read) results[i]->read(mxnet_ndarray, 0, buffer_size);
    }
  }
}
}  // namespace ngraph_bridge

#endif  // MXNET_NGRAPH_NGRAPH_NNVM_UTILS_H_
