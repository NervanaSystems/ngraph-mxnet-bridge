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
#include <vector>

#include "../../../../src/operator/nn/convolution-inl.h"
#include "convolution.h"

#include <ngraph/builder/quantization.hpp>
#include "../ngraph_emitter.h"
#include "../ngraph_emitter_utils.h"
#include "../ngraph_nnvm_ops.h"
#include "../ngraph_sgcompiler_utils.h"
#include "../ngraph_utils.h"

namespace ngraph_bridge {

struct ConvInputs {
  NgraphNodePtr data;
  NgraphNodePtr filter;
  NgraphNodePtr bias;
  ngraph::CoordinateDiff pad;
  ngraph::Strides stride;
  ngraph::Strides dilate;
  size_t groups;
};

ConvInputs get_conv_inputs(Emitter* emitter, const NodePtr& node,
                           const mxnet::op::ConvolutionParam& param) {
  ConvInputs conv_inputs;
  conv_inputs.data = emitter->op_map_[node->inputs_[0]];
  conv_inputs.filter = emitter->op_map_[node->inputs_[1]];
  conv_inputs.bias =
      param.no_bias ? nullptr : emitter->op_map_[node->inputs_[2]];
  conv_inputs.pad = ngraph::CoordinateDiff{param.pad.begin(), param.pad.end()};
  conv_inputs.stride =
      ngraph::Strides{param.stride.begin(), param.stride.end()};
  conv_inputs.dilate =
      ngraph::Strides{param.dilate.begin(), param.dilate.end()};
  conv_inputs.groups = param.num_group;
  return conv_inputs;
}
NgraphNodePtr create_convolution(Emitter* emitter, const NodePtr& node) {
  const auto& param =
      nnvm::get<mxnet::op::ConvolutionParam>(node->orig_node_->attrs.parsed);
  auto conv_inputs = get_conv_inputs(emitter, node, param);

  auto dshape = conv_inputs.data->get_shape();
  auto fshape = conv_inputs.filter->get_shape();

  NgraphNodePtr convolution = nullptr;
  if (conv_inputs.groups == 1) {
    convolution = std::make_shared<ngraph::op::Convolution>(
        conv_inputs.data, conv_inputs.filter, conv_inputs.stride,
        conv_inputs.dilate, conv_inputs.pad, conv_inputs.pad);
  } else {
    std::vector<NgraphNodePtr> convolutions(conv_inputs.groups);
    for (size_t g = 0; g < conv_inputs.groups; ++g) {
      // slice data on channel_in
      size_t data_slice_step = dshape[1] / conv_inputs.groups;
      size_t filter_slice_step = fshape[0] / conv_inputs.groups;
      auto data_slice = slice_data_on_axis(
          conv_inputs.data, g * data_slice_step, data_slice_step, 1, false);
      auto filter_slice =
          slice_data_on_axis(conv_inputs.filter, g * filter_slice_step,
                             filter_slice_step, 0, false);
      // convolve sliced data and filter
      // N, channel_out/groups, d'1,...,d'n
      convolutions[g] = std::make_shared<ngraph::op::Convolution>(
          data_slice, filter_slice, conv_inputs.stride, conv_inputs.dilate,
          conv_inputs.pad, conv_inputs.pad);
    }

    // concatenate convolutions on channel_out
    // N, channel_out, d'1,...,d'n
    convolution = std::make_shared<ngraph::op::Concat>(convolutions, 1);
  }

  // no bias param, return
  if (!conv_inputs.bias) {
    return convolution;
  }

  // 1, channel_out, 1,...,1
  ngraph::Shape bias_shape(fshape.size(), 1);
  bias_shape[1] = fshape[0];

  ngraph::AxisVector order(1, 0);
  auto bias_reshape = std::make_shared<ngraph::op::Reshape>(conv_inputs.bias,
                                                            order, bias_shape);

  return ngraph::builder::make_with_numpy_broadcast<ngraph::op::Add>(
      convolution, bias_reshape);
}

NgraphNodePtr create_quantized_convolution(Emitter* emitter,
                                           const NodePtr& node) {
  const auto& param_ = nnvm::get<mxnet::op::MKLDNNConvFusionParam>(
      node->orig_node_->attrs.parsed);
  auto& full_conv_param = param_.full_conv_param;
  auto& mkldnn_param = param_.full_conv_param.mkldnn_param;
  auto& conv_param = param_.full_conv_param.conv_param;
  auto bn_param = param_.bn_param.get();
  auto conv_inputs = get_conv_inputs(emitter, node, conv_param);
  if (conv_inputs.groups != 1 || !conv_inputs_bias) {
    throw std::runtime_error(
        "nobias or group>1 not supported by quantized_conv for now");
  }
  auto fshape = conv_inputs.filter->get_shape();
  auto data_n = emitter->op_map_[node->inputs_[3]];
  auto data_m = emitter->op_map_[node->inputs_[4]];
  auto filter_n = emitter->op_map_[node->inputs_[5]];
  auto filter_m = emitter->op_map_[node->inputs_[6]];
  NgraphNodePtr bias, bias_n, bias_m;
  if (conv_inputs.bias) {
    bias_n = emitter->op_map_[node->inputs_[3]];
    bias_m = emitter->op_map_[node->inputs_[4]];
    ngraph::Shape bias_shape(fshape.size(), 1);
    bias_shape[1] = fshape[0];
    ngraph::AxisVector order(1, 0);
    bias = std::make_shared<ngraph::op::Reshape>(conv_inputs.bias, order,
                                                 bias_shape);
  }
  auto min = makeConstant(ngraph::element::f32, ngraph::Shape{},
                          std::to_string(mkldnn_param.min_calib_range.value()));
  auto max = makeConstant(ngraph::element::f32, ngraph::Shape{},
                          std::to_string(mkldnn_param.max_calib_range.value()));

  auto conv = ngraph::builder::ScaledQuantizedConvolutionBias(
      conv_inputs.data, conv_inputs.filter, bias, conv_inputs.stride,
      conv_inputs.dilate, conv_inputs.pad, conv_inputs.pad, conv_inputs.dilate,
      data_n, data_m, filter_n, filter_m, min, max);
  return conv;
}
}
