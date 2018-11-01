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
#include "ngraph_nnvm_utils.h"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "../../../src/operator/operator_common.h"
#include "../../../src/operator/subgraph/common.h"
#include "ngraph_compiler.h"
#include "ngraph_nnvm_ops.h"
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
  for (size_t i = 0; i < cached_aux_count; ++i) {
    auto buffer_size = results[i]->get_size_in_bytes();

    void *mxnet_ndarray = inputs[graph->cached_aux_positions[mode][i] + offset]
                              .storage_handle()
                              .dptr;
    results[i]->read(mxnet_ndarray, 0, buffer_size);
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
  bool is_train = ctx.is_train;
  if (!graph->need_grad) {
    is_train = false;
  }

  int mode = static_cast<int>(GraphExeMode::kInfer);
  if (is_train) {
    mode = static_cast<int>(GraphExeMode::kTrain);
    graph->forward_train_computed = true;
  }

  compile_if_needed(graph, mode);

  auto placeholders = get_tensors(
      inputs, backend, graph->bool_nodes_[mode][(int)(NodeReferences::kForwardInput)],
      graph->scalar_nodes_[mode][(int)(NodeReferences::kForwardInput)], nullptr,
      graph->is_reuse_mem);

  // for outputs we need to comply with req
  TensorVector results;
  if (is_train) {
    results = get_tensors(
        outputs, backend, graph->bool_nodes_[mode][(int)(NodeReferences::kForwardOutput)],
        graph->scalar_nodes_[mode][(int)(NodeReferences::kForwardOutput)], &req,
        graph->is_reuse_mem);
  } else {
    results = get_tensors(
        std::vector<mxnet::NDArray>(outputs.begin(),
                                    outputs.begin() + graph->num_outputs_),
        backend, graph->bool_nodes_[mode][(int)(NodeReferences::kForwardOutput)],
        graph->scalar_nodes_[mode][(int)(NodeReferences::kForwardOutput)], &req,
        graph->is_reuse_mem);
  }

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
  if (!graph->need_grad) {
    return;
  }

  // only expect backward is called in training mode
  auto backend = graph->get_backend();

  const int mode = static_cast<int>(GraphExeMode::kTrain);
  compile_if_needed(graph, mode);

  auto input_tvs = get_tensors(
      inputs, backend, graph->bool_nodes_[mode][(int)(NodeReferences::kBackwardInput)],
      graph->scalar_nodes_[mode][(int)(NodeReferences::kBackwardInput)], nullptr,
      graph->is_reuse_mem);

  size_t adjoints = 0;
  if (!graph->zero_grad) {
    adjoints = graph->num_adjoints_;
  }

  TensorVector placeholders;
  TensorVector aux_results;
  size_t i = 0;
  size_t inputs_size = adjoints + graph->inputs_.size();
  size_t aux_size = graph->cached_aux_positions[mode].size();
  for (const auto &tv : input_tvs) {
    if ((i >= inputs_size) && (i < (aux_size + inputs_size))) {
      aux_results.push_back(tv);
    } else {
      placeholders.push_back(tv);
    }
    ++i;
  }
  for (size_t i = 0; i < graph->num_adjoints_; ++i) {
    if (graph->zero_grad || graph->is_loss[i]) {
      placeholders.insert(
          placeholders.begin() + i,
          backend->create_tensor(getType(graph->outputs_[i]->dtype_),
                                 TShape_to_NShape(graph->outputs_[i]->shape_)));
    }
  }
  auto results = get_tensors(
      outputs, backend, graph->bool_nodes_[mode][(int)(NodeReferences::kBackwardOutput)],
      graph->scalar_nodes_[mode][(int)(NodeReferences::kBackwardOutput)], &req,
      graph->is_reuse_mem);

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

// TODO: temp workaround for int8, use DO's operator structs
namespace mxnet {
namespace op {
static inline size_t GetInSumIndex(const MKLDNNConvFusionParam &param) {
  return 2 + (param.full_conv_param.conv_param.no_bias ? 0 : 1) +
         (param.full_conv_param.mkldnn_param.with_bn ? 4 : 0);
}
static uint32_t SgMKLDNNConvNumInputs(const NodeAttrs &attrs) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  auto num_input = DefaultSubgraphOpNumInputs(attrs);
  if (param.full_conv_param.mkldnn_param.quantized)
    return num_input + 2 + param.full_conv_param.mkldnn_param.with_sum ? 2 : 0;
  else
    return num_input;
}
DMLC_REGISTER_PARAMETER(MKLDNNConvParam);
static void SgMKLDNNConvParamParser(nnvm::NodeAttrs *attrs) {
  MKLDNNConvFusionParam param_;
  try {
    param_.full_conv_param.mkldnn_param.Init(attrs->dict);
  } catch (const dmlc::ParamError &e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto &k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  auto subgraph_sym = attrs->subgraphs[0];
  DFSVisit(subgraph_sym->outputs, [&](const nnvm::NodePtr &node) {
    if (node->is_variable()) return;
    auto &node_name = node->op()->name;
    if (node_name == "BatchNorm") {
      CHECK_EQ(param_.full_conv_param.mkldnn_param.with_bn, true);
      CHECK(param_.bn_param.get() == nullptr);
      param_.bn_param = std::make_shared<BatchNormParam>(
          nnvm::get<BatchNormParam>(node->attrs.parsed));
    } else if (node_name == "Convolution") {
      param_.full_conv_param.conv_param =
          nnvm::get<ConvolutionParam>(node->attrs.parsed);
    }
  });
  attrs->parsed = std::move(param_);
}

static std::vector<std::string> SgMKLDNNConvListInputNames(
    const NodeAttrs &attrs) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  std::vector<std::string> input_names = DefaultSubgraphOpListInputs(attrs);
  if (param.full_conv_param.mkldnn_param.quantized) {
    input_names.emplace_back("data_min");
    input_names.emplace_back("data_max");
    if (param.full_conv_param.mkldnn_param.with_sum) {
      input_names.emplace_back("sum_min");
      input_names.emplace_back("sum_max");
    }
  }
  return input_names;
}
template <typename DType>
static void FilterMinMaxIndice(const MKLDNNConvParam &mkldnn_param,
                               std::vector<DType> *in_shapes,
                               std::vector<DType> *out_shapes,
                               std::vector<DType> *base_in_shapes,
                               std::vector<DType> *base_out_shapes,
                               std::unordered_set<size_t> *minmax_indice) {
  base_out_shapes->push_back(out_shapes->at(0));
  size_t last = in_shapes->size() - 1;
  if (mkldnn_param.with_sum) {
    minmax_indice->insert(last);
    minmax_indice->insert(last - 1);
    minmax_indice->insert(last - 2);
    minmax_indice->insert(last - 3);
    *base_in_shapes =
        std::vector<DType>(in_shapes->begin(), in_shapes->end() - 4);
  } else {
    minmax_indice->insert(last);
    minmax_indice->insert(last - 1);
    *base_in_shapes =
        std::vector<DType>(in_shapes->begin(), in_shapes->end() - 2);
  }
}
static bool SgMKLDNNConvInferShape(const nnvm::NodeAttrs &attrs,
                                   std::vector<TShape> *in_shapes,
                                   std::vector<TShape> *out_shapes) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.mkldnn_param.quantized) {
    std::unordered_set<size_t> minmax_indice;
    std::vector<TShape> base_in_shapes;
    std::vector<TShape> base_out_shapes;

    FilterMinMaxIndice<TShape>(param.full_conv_param.mkldnn_param, in_shapes,
                               out_shapes, &base_in_shapes, &base_out_shapes,
                               &minmax_indice);
    bool result =
        DefaultSubgraphOpShape(attrs, &base_in_shapes, &base_out_shapes);
    size_t base_idx = 0;
    for (size_t i = 0; i < in_shapes->size(); ++i) {
      if (minmax_indice.count(i)) {
        SHAPE_ASSIGN_CHECK(*in_shapes, i, Shape1(1));
      } else {
        in_shapes->at(i) = base_in_shapes[base_idx++];
      }
    }
    out_shapes->at(0) = base_out_shapes[0];
    SHAPE_ASSIGN_CHECK(*out_shapes, 1, Shape1(1));
    SHAPE_ASSIGN_CHECK(*out_shapes, 2, Shape1(1));
    return result;
  } else {
    return DefaultSubgraphOpShape(attrs, in_shapes, out_shapes);
  }
}

static bool SgMKLDNNConvInferType(const nnvm::NodeAttrs &attrs,
                                  std::vector<int> *in_types,
                                  std::vector<int> *out_types) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.mkldnn_param.quantized) {
    std::unordered_set<size_t> minmax_indice;
    std::vector<int> base_in_types;
    std::vector<int> base_out_types;
    FilterMinMaxIndice<int>(param.full_conv_param.mkldnn_param, in_types,
                            out_types, &base_in_types, &base_out_types,
                            &minmax_indice);
    // Override data type to fp32 for default infer type as bn doesn't support
    // uint8.
    int orig_data = base_in_types[0];
    base_in_types[0] = mshadow::kFloat32;
    int orig_sum = base_in_types[0];
    if (param.full_conv_param.mkldnn_param.with_sum) {
      auto sum_index = GetInSumIndex(param);
      orig_sum = base_in_types[sum_index];
      base_in_types[sum_index] = mshadow::kFloat32;
    }
    bool result = DefaultSubgraphOpType(attrs, &base_in_types, &base_out_types);
    base_in_types[0] = orig_data;
    if (param.full_conv_param.mkldnn_param.with_sum) {
      auto sum_index = GetInSumIndex(param);
      base_in_types[sum_index] = orig_sum;
    }
    size_t base_idx = 0;
    for (size_t i = 0; i < in_types->size(); ++i) {
      if (minmax_indice.count(i)) {
        TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
      } else {
        in_types->at(i) = base_in_types[base_idx++];
      }
    }
    if (param.full_conv_param.mkldnn_param.min_calib_range.has_value() &&
        param.full_conv_param.mkldnn_param.max_calib_range.has_value()) {
      if (IsOutputUInt8(param.full_conv_param.mkldnn_param)) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kUint8);
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);
      }
    } else {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);
    }

    TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);
    return result;
  } else {
    return DefaultSubgraphOpType(attrs, in_types, out_types);
  }
}

static bool SgMKLDNNConvOpStorageType(const nnvm::NodeAttrs &attrs,
                                      const int dev_mask,
                                      DispatchMode *dispatch_mode,
                                      std::vector<int> *in_stypes,
                                      std::vector<int> *out_stypes) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.mkldnn_param.quantized) {
    std::unordered_set<size_t> minmax_indice;
    std::vector<int> base_in_stypes;
    std::vector<int> base_out_stypes;
    FilterMinMaxIndice<int>(param.full_conv_param.mkldnn_param, in_stypes,
                            out_stypes, &base_in_stypes, &base_out_stypes,
                            &minmax_indice);
    bool result = DefaultSubgraphOpStorageType(
        attrs, dev_mask, dispatch_mode, &base_in_stypes, &base_out_stypes);
    size_t base_idx = 0;
    for (size_t i = 0; i < in_stypes->size(); ++i) {
      if (minmax_indice.count(i)) {
        type_assign(&in_stypes->at(i), mxnet::kDefaultStorage);
      } else {
        in_stypes->at(i) = base_in_stypes[base_idx++];
      }
    }
    out_stypes->at(0) = base_out_stypes[0];
    type_assign(&out_stypes->at(1), mxnet::kDefaultStorage);
    type_assign(&out_stypes->at(2), mxnet::kDefaultStorage);
    return result;
  } else {
    return DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode,
                                        in_stypes, out_stypes);
  }
}
static std::vector<std::string> SgMKLDNNConvListOutputNames(
    const NodeAttrs &attrs) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.mkldnn_param.quantized)
    return std::vector<std::string>{"output", "output_min", "output_max"};
  else
    return std::vector<std::string>{"output"};
}
std::vector<std::pair<int, int>> SgMKLDNNConvInplaceOption(
    const NodeAttrs &attrs) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.mkldnn_param.with_sum) {
    return std::vector<std::pair<int, int>>{{GetInSumIndex(param), 0}};
  } else {
    return std::vector<std::pair<int, int>>();
  }
}

NNVM_REGISTER_OP(_sg_mkldnn_conv)
    .describe(R"code(_sg_mkldnn_conv)code" ADD_FILELINE)
    .set_num_inputs(SgMKLDNNConvNumInputs)
    .set_num_outputs([](const NodeAttrs &attrs) {
      auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
      return param.full_conv_param.mkldnn_param.quantized ? 3 : 1;
    })
    .set_attr_parser(SgMKLDNNConvParamParser)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     SgMKLDNNConvListInputNames)
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      SgMKLDNNConvListOutputNames)
    .set_attr<nnvm::FInferShape>("FInferShape", SgMKLDNNConvInferShape)
    .set_attr<nnvm::FInferType>("FInferType", SgMKLDNNConvInferType)
    .set_attr<FInferStorageType>("FInferStorageType", SgMKLDNNConvOpStorageType)
    .set_attr<nnvm::FMutateInputs>("FMutateInputs",
                                   DefaultSubgraphOpMutableInputs)
    .set_attr<std::string>("key_var_num_args", "num_args")
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    SgMKLDNNConvInplaceOption);
}
}  // namespace mxnet op
