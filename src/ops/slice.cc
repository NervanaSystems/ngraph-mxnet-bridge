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

#include "../../../../src/operator/tensor/matrix_op-inl.h"
#include "../ngraph_sgcompiler_utils.h"
#include "slice.h"

namespace ngraph_bridge {
NgraphNodePtr get_slice(const NgraphNodePtr& node,
                        const nnvm::Tuple<dmlc::optional<int>>& param_begin,
                        const nnvm::Tuple<dmlc::optional<int>>& param_end,
                        const nnvm::Tuple<dmlc::optional<int>>& param_step) {
  mxnet::TShape tshape = NShape_to_TShape(node->get_shape());
  ngraph::Coordinate ng_begin, ng_end, ng_step;
  ngraph::AxisSet reverse_axes;
  const bool reverse = std::any_of(param_step.begin(), param_step.end(),
                                   [](const dmlc::optional<int>& s) {
                                     return s.has_value() && s.value() < 0;
                                   });

  for (mxnet::index_t i = 0; i < param_begin.ndim(); ++i) {
    const int len = tshape[i];

    int s = 1;
    if (param_step[i].has_value()) {
      s = param_step[i].value();
      if (s == 0) {
        s = 1;
      }
    }

    int b = 0;
    if (param_begin[i].has_value()) {
      b = param_begin[i].value();
      if (b < 0) {
        b += len;
      }
    } else if (s < 0) {
      b = len - 1;
    }

    int e = len;
    if (param_end[i].has_value()) {
      e = param_end[i].value();
      if (e < 0) {
        e += len;
      }
    } else if (s < 0) {
      e = -1;
    }

    if (s < 0) {
      s = abs(s);
      reverse_axes.insert(i);
      int tempb = b;
      int last = b;
      while (last > e + s) {
        last -= s;
      }
      b = last;
      e = tempb + 1;
    }

    ng_begin.push_back(b);
    ng_end.push_back(e);
    ng_step.push_back(s);
  }

  for (mxnet::index_t i = param_begin.ndim(); i < tshape.ndim(); ++i) {
    ng_begin.push_back(0);
    ng_end.push_back(tshape[i]);
    ng_step.push_back(1);
  }

  NgraphNodePtr slice;
  if (reverse) {
    slice = std::make_shared<ngraph::op::Reverse>(
        std::make_shared<ngraph::op::Slice>(node, ng_begin, ng_end, ng_step),
        ngraph::AxisSet{reverse_axes});
  } else {
    slice =
        std::make_shared<ngraph::op::Slice>(node, ng_begin, ng_end, ng_step);
  }
  return slice;
}

NgraphNodePtr create_slice_op(const NgraphNodePtr& node,
                              const nnvm::NodeAttrs& attrs) {
  const mxnet::op::SliceParam& param =
      nnvm::get<mxnet::op::SliceParam>(attrs.parsed);
  return get_slice(node, param.begin, param.end, param.step);
}

NgraphNodePtr create_slice_like_op(const NgraphNodePtr& node,
                                   const NgraphNodePtr& out,
                                   const nnvm::NodeAttrs& attrs) {
  const mxnet::op::SliceLikeParam& param =
      nnvm::get<mxnet::op::SliceLikeParam>(attrs.parsed);
  mxnet::TShape shape = NShape_to_TShape(node->get_shape());
  mxnet::TShape tshape = NShape_to_TShape(out->get_shape());
  nnvm::Tuple<dmlc::optional<int>> param_begin;
  nnvm::Tuple<dmlc::optional<int>> param_end;
  nnvm::Tuple<dmlc::optional<int>> param_step;
  mxnet::op::SliceLikeInferRanges(shape, tshape, param.axes, &param_begin,
                                  &param_end, &param_step);
  return get_slice(node, param_begin, param_end, param_step);
}
}  // namespace ngraph_bridge
