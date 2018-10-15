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
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mxnet/operator.h>
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/symbolic.h>

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "../../../src/operator/operator_common.h"
#include "ngraph_compiler.h"
#include "ngraph_nnvm_ops.h"
#include "ngraph_nnvm_utils.h"
#include "ngraph_sgcompiler.h"
#include "ngraph_utils.h"

namespace ngraph_bridge {

#if MXNET_USE_CUDA
#define NGRAPH_TRANSFORMERS \
  { "cpu", "gpu" }
#else
#define NGRAPH_TRANSFORMERS \
  { "cpu" }
#endif

void update_aux_vals(const std::shared_ptr<Graph> &graph,
                     const TensorVector &results,
                     const std::vector<mxnet::NDArray> &inputs, const int mode,
                     const int offset = 0) {
  const size_t cached_aux_count = graph->cached_aux_positions[mode].size();
  if (cached_aux_count > 0) {
    std::vector<mxnet::OpReqType> aux_req;
    std::vector<mxnet::NDArray> aux_outs;

    for (size_t i = 0; i < cached_aux_count; ++i) {
      aux_outs.push_back(inputs[graph->cached_aux_positions[mode][i] + offset]);
      aux_req.push_back(mxnet::kWriteTo);
    }

    result_to_NDArray(results, aux_req, aux_outs, true);
  }
}

void compile_if_needed(std::shared_ptr<Graph> graph, int mode) {
  if (mode == static_cast<int>(GraphExeMode::kTrain)) {
    if (graph->ngraph_forward[mode] == nullptr) {
      CompileForwardBackward(graph, graph->fprop_cache->fprop,
                             graph->fprop_cache->bprop, GraphExeMode::kTrain,
                             *(graph->fprop_cache));
    }
  }
}

// function for computing forward on ngraph
void compute_forward(const mxnet::OpContext &ctx, std::shared_ptr<Graph> graph,
                     const std::vector<mxnet::NDArray> &inputs,
                     const std::vector<mxnet::OpReqType> &req,
                     const std::vector<mxnet::NDArray> &outputs) {
  auto backend = graph->get_backend();
  auto placeholders =
      get_tensors(inputs, backend, nullptr, graph->is_reuse_mem);
  // for outputs we need to comply with req
  TensorVector results;
  if (ctx.is_train) {
    results = get_tensors(outputs, backend, &req, graph->is_reuse_mem);
  } else {
    std::vector<mxnet::NDArray> inference_outputs;
    for (size_t i = 0; i < graph->num_outputs_; ++i) {
      inference_outputs.push_back(outputs[i]);
    }
    results = get_tensors(inference_outputs, backend, &req, graph->is_reuse_mem);
  }

  int mode = static_cast<int>(GraphExeMode::kInfer);
  if (ctx.is_train) {
    mode = static_cast<int>(GraphExeMode::kTrain);
    graph->forward_train_computed = true;
  }
  compile_if_needed(graph, mode);

  if (mode == static_cast<int>(GraphExeMode::kTrain)) {
    for (auto &tv : placeholders) {
      tv->set_stale(true);
    }
  }

  backend->call(graph->ngraph_forward[mode], results, placeholders);

  result_to_NDArray(results, req, outputs, !graph->is_reuse_mem);

  if (mode == static_cast<int>(GraphExeMode::kInfer)) {
    for (size_t i = 0; i < placeholders.size(); ++i) {
      if (graph->input_is_weight_[i]) {
        placeholders[i]->set_stale(false);
      }
    }
    TensorVector aux_results;
    aux_results.insert(aux_results.end(), results.begin() + graph->num_outputs_,
                       results.begin() + graph->num_outputs_ +
                           graph->cached_aux_positions[mode].size());
    update_aux_vals(graph, aux_results, inputs, mode);
  }
}
// function for computing backward on ngraph
void compute_backward(const mxnet::OpContext &ctx, std::shared_ptr<Graph> graph,
                      const std::vector<mxnet::NDArray> &inputs,
                      const std::vector<mxnet::OpReqType> &req,
                      const std::vector<mxnet::NDArray> &outputs) {
  // only expect backward is called in training mode
  auto backend = graph->get_backend();

  const int mode = static_cast<int>(GraphExeMode::kTrain);
  compile_if_needed(graph, mode);

  TensorVector placeholders;    
  TensorVector aux_results;
  auto input_tvs = get_tensors(inputs, backend, &req, graph->is_reuse_mem);

  size_t adjoints = 0;
  if (!graph->zero_grad) {
    adjoints = graph->num_adjoints_;
  }
  auto end_of_adjoints = input_tvs.begin() + adjoints + graph->inputs_.size();
  auto end_of_aux = end_of_adjoints + graph->cached_aux_positions[mode].size();
  placeholders.insert(placeholders.end(), input_tvs.begin(), end_of_adjoints);
  aux_results.insert(aux_results.end(), end_of_adjoints, end_of_aux);
  placeholders.insert(placeholders.end(), end_of_aux, input_tvs.end());

  if (graph->zero_grad) {
    for (size_t i = 0; i < graph->num_adjoints_; ++i) {
      // TODO(mbrookahrt): don't bprop graph if it's zerograd?
      placeholders.insert(
          placeholders.begin(),
          backend->create_tensor(getType(graph->outputs_[i]->dtype_),
                                 TShape_to_NShape(graph->outputs_[i]->shape_)));
    }
  }

  auto results = get_tensors(outputs, backend, &req, graph->is_reuse_mem);

  CHECK(graph->ngraph_backward[mode]);
  backend->call(graph->ngraph_backward[mode], results, placeholders);
  // reset the forward training compute flag to ensure backward always have
  // updated data from forward
  graph->forward_train_computed = false;
  result_to_NDArray(results, req, outputs, !graph->is_reuse_mem);

  // overwrite aux data if they exist
  // aux result outputs mapped to inputs
  update_aux_vals(graph, aux_results, inputs, mode, graph->num_adjoints_);
}

// check if last node in graph is an op that doesnt need head-gradient
bool check_zero_grad(const std::shared_ptr<Graph> &graph) {
  auto size = graph->ngraph_forward[0]->get_ops().size();
  if (size < 1) return false;

  // if all of the outputs of the graph don't need gradient calculation,
  // don't autodiff this graph. Otherwise, do.
  for (auto node : graph->outputs_) {
    if (node->operation_ == "SoftmaxOutput") {
      if (get_default(node, "out_grad", false)) {
        return false;
      }
    } else if (ops_no_head_grad.count(node->operation_) == 0) {
      return false;
    }
  }

  return true;
}

}  // namespace ngraph_bridge
