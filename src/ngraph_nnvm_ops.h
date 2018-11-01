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

#ifndef MXNET_NGRAPH_NGRAPH_NNVM_OPS_H_
#define MXNET_NGRAPH_NGRAPH_NNVM_OPS_H_

#include <mxnet/ndarray.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/op.h>
#include "../../../src/operator/nn/batch_norm-inl.h"
#include "../../../src/operator/nn/convolution-inl.h"

#include <string>
#include <vector>

#include "ngraph_graph.h"

namespace ngraph_bridge {
// function for computing forward on ngraph
void compute_forward(const mxnet::OpContext& ctx, std::shared_ptr<Graph> graph,
                     const std::vector<mxnet::NDArray>& inputs,
                     const std::vector<mxnet::OpReqType>& req,
                     const std::vector<mxnet::NDArray>& outputs);
// function for computing backward on ngraph
void compute_backward(const mxnet::OpContext& ctx, std::shared_ptr<Graph> graph,
                      const std::vector<mxnet::NDArray>& inputs,
                      const std::vector<mxnet::OpReqType>& req,
                      const std::vector<mxnet::NDArray>& outputs);

// dummy parameter struct to match mxnet API
struct NGraphParam {
  std::vector<std::string> arguments;
  std::vector<std::string> aux_states;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  void Init(const nnvm::NodeAttrs& attrs) {}
  std::shared_ptr<ngraph_bridge::Graph> g;
};
bool check_zero_grad(const std::shared_ptr<Graph>& graph);
}  // namespace ngraph_bridge
namespace mxnet {
namespace op {
struct MKLDNNConvParam : public dmlc::Parameter<MKLDNNConvParam> {
  bool with_bn;
  bool with_relu;
  bool with_sum;
  bool with_postsum_relu;
  bool quantized;
  bool weight_channelwise_scale;

  dmlc::optional<float>
      min_calib_range;  // min float value calculated from calibration dataset
  dmlc::optional<float>
      max_calib_range;  // max float value calculated from calibration dataset

  DMLC_DECLARE_PARAMETER(MKLDNNConvParam) {
    DMLC_DECLARE_FIELD(with_bn).set_default(false).describe(
        "Add post batchnorm.");
    DMLC_DECLARE_FIELD(with_relu).set_default(false).describe("Add post relu");
    DMLC_DECLARE_FIELD(with_sum).set_default(false).describe("Add post sum");
    DMLC_DECLARE_FIELD(with_postsum_relu)
        .set_default(false)
        .describe("Add post relu after sum");
    DMLC_DECLARE_FIELD(quantized).set_default(false).describe(
        "enable quantization");
    DMLC_DECLARE_FIELD(weight_channelwise_scale)
        .set_default(true)
        .describe("Quantize weight with channel wise scales.");
    DMLC_DECLARE_FIELD(min_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The minimum scalar value in the form of float32 obtained "
            "through calibration. If present, it will be used to by "
            "quantized convolution op to calculate primitive scale");
    DMLC_DECLARE_FIELD(max_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The maximum scalar value in the form of float32 obtained "
            "through calibration. If present, it will be used to by "
            "quantized convolution op to calculate primitive scale");
  }
};
struct MKLDNNConvFullParam {
  ConvolutionParam conv_param;
  MKLDNNConvParam mkldnn_param;
  float sum_scale;
  std::vector<float> requantize_scales;
};

static inline bool IsOutputUInt8(const MKLDNNConvParam& mkldnn_param) {
  return ((!mkldnn_param.with_sum) && mkldnn_param.with_relu) ||
         mkldnn_param.with_postsum_relu;
}
struct MKLDNNConvFusionParam {
  MKLDNNConvFullParam full_conv_param;
  std::shared_ptr<BatchNormParam> bn_param;
};
}
}

#endif  // MXNET_NGRAPH_NGRAPH_NNVM_OPS_H_
