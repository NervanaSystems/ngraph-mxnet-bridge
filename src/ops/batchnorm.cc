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

#include "batchnorm.h"

#include <ngraph/op/get_output_element.hpp>

#include "../ngraph_emitter.h"
#include "../ngraph_sgcompiler_utils.h"
#include "../ngraph_utils.h"

using std::make_shared;
using ngraph::builder::make_with_numpy_broadcast;

namespace ngraph_bridge {

/// Create a subgraph that computes BatchNorm _without_ using the nGraph
/// BatchNorm operator.
static NgraphNodePtr create_batchnorm_basic_computation_nodes(
    const NgraphNodePtr& ng_mean, const NgraphNodePtr& ng_variance,
    const NgraphNodePtr& ng_in_data, const ngraph::AxisVector& axes,
    const float epsilon, const NgraphNodePtr& ng_maybe_gamma,
    const NgraphNodePtr& ng_beta, bool is_batchnorm = true) {
  const ngraph::Shape& batch_data_shape = ng_in_data->get_shape();
  const size_t batch_data_rank = batch_data_shape.size();

  const ngraph::element::Type et = ng_in_data->get_element_type();
  check(ng_beta->get_element_type() == et);

  // Get our input tensors / constants into the required shape...
  const ngraph::Shape reduced_axes_shape =
      get_reduced_axes_shape(batch_data_shape, axes);

  const NgraphNodePtr ng_mean_shaped = std::make_shared<ngraph::op::Reshape>(
      ng_mean, pyrange(ng_mean->get_shape().size()), reduced_axes_shape);

  const NgraphNodePtr ng_var_shaped = std::make_shared<ngraph::op::Reshape>(
      ng_variance, pyrange(ng_variance->get_shape().size()),
      reduced_axes_shape);

  const NgraphNodePtr ng_epsilon_shaped =
      makeConstant(et, reduced_axes_shape, epsilon);

  size_t channel_axis;
  if (axes.size() == 1 && is_batchnorm) {
    channel_axis = axes[0];
  } else if (axes.size() == batch_data_shape.size() - 1) {
    auto all_axes = pyrange(batch_data_shape.size());
    auto channel_axes = ngraph::Shape(1);
    std::set_difference(all_axes.begin(), all_axes.end(), axes.begin(),
                        axes.end(), channel_axes.begin());
    std::cout << "all_axes" << all_axes << std::endl;
    std::cout << "axes" << axes << std::endl;
    std::cout << "channel_axes" << channel_axes << std::endl;
    channel_axis = channel_axes[0];
  } else {
    check(axes.size() == batch_data_shape.size() - 1);
  }

  const NgraphNodePtr ng_beta_shaped =
      ensure_vector_plus_axes_shape(ng_beta, batch_data_rank, channel_axis);

  // Create the computation nodes...
  const NgraphNodePtr denom =
      make_shared<ngraph::op::Sqrt>(ng_var_shaped + ng_epsilon_shaped);

  const NgraphNodePtr numerator =
      make_with_numpy_broadcast<ngraph::op::Subtract>(ng_in_data,
                                                      ng_mean_shaped);

  const NgraphNodePtr ng_post_simply_normalized =
      make_with_numpy_broadcast<ngraph::op::Divide>(numerator, denom);

  NgraphNodePtr ng_post_gamma_result;
  if (ng_maybe_gamma) {
    const NgraphNodePtr ng_gamma_shaped = ensure_vector_plus_axes_shape(
        ng_maybe_gamma, batch_data_rank, channel_axis);
    std::cout << channel_axis << std::endl;
    std::cout << "beta " << ng_beta_shaped->get_shape() << std::endl;
    std::cout << "gamma " << ng_gamma_shaped->get_shape() << std::endl;

    ng_post_gamma_result = make_with_numpy_broadcast<ngraph::op::Multiply>(
        ng_post_simply_normalized, ng_gamma_shaped);
  } else {
    ng_post_gamma_result = ng_post_simply_normalized;
  }

  const NgraphNodePtr ng_post_beta_result =
      make_with_numpy_broadcast<ngraph::op::Add>(ng_post_gamma_result,
                                                 ng_beta_shaped);

  return ng_post_beta_result;
}

std::tuple<NgraphNodePtr, NgraphNodePtr, NgraphNodePtr>
create_normalization_subgraph(const float epsilon,
                              const NgraphNodePtr ng_maybe_gamma,
                              const NgraphNodePtr ng_beta,
                              const NgraphNodePtr ng_in_data,
                              const ngraph::AxisVector& axes,
                              bool is_batchnorm = true) {
  const ngraph::Shape& batch_data_shape = ng_in_data->get_shape();
  const size_t batch_data_rank = batch_data_shape.size();

  const ngraph::element::Type et = ng_in_data->get_element_type();
  check(ng_beta->get_element_type() == et);

  const NgraphNodePtr ng_batch_means =
      Emitter::ReduceAxes(ng_in_data, axes, true, true, ngraph::builder::mean);

  const NgraphNodePtr ng_batch_variances =
      Emitter::ReduceAxes(ng_in_data, axes, true, true,
                          [](const std::shared_ptr<ngraph::Node>& node,
                             const ngraph::AxisSet& axes) {
                            return ngraph::builder::variance(node, axes);
                          });

  const NgraphNodePtr ng_normalized_batch =
      create_batchnorm_basic_computation_nodes(
          ng_batch_means, ng_batch_variances, ng_in_data, axes, epsilon,
          ng_maybe_gamma, ng_beta, is_batchnorm);
  NgraphNodePtr ng_batch_means_vector_shaped;
  NgraphNodePtr ng_batch_variances_vector_shaped;
  if (axes.size() == 1) {
    ng_batch_means_vector_shaped = ensure_vector_only_shape(ng_batch_means);
    ng_batch_variances_vector_shaped =
        ensure_vector_only_shape(ng_batch_variances);
  } else {
    ng_batch_means_vector_shaped = ng_batch_means;
    ng_batch_variances_vector_shaped = ng_batch_variances;
  }

  return std::tuple<NgraphNodePtr, NgraphNodePtr, NgraphNodePtr>{
      ng_normalized_batch, ng_batch_means_vector_shaped,
      ng_batch_variances_vector_shaped};
}

NgraphNodePtr create_normalization_subgraph(
    const float epsilon, const NgraphNodePtr ng_maybe_gamma,
    const NgraphNodePtr ng_beta, const NgraphNodePtr ng_in_data,
    const NgraphNodePtr ng_moving_mean, const NgraphNodePtr ng_moving_var,
    const ngraph::AxisVector& axes, bool is_batchnorm = true) {
  return create_batchnorm_basic_computation_nodes(
      ng_moving_mean, ng_moving_var, ng_in_data, axes, epsilon, ng_maybe_gamma,
      ng_beta, is_batchnorm);
}

std::tuple<NgraphNodePtr, NgraphNodePtr, NgraphNodePtr>
create_batchnorm_training_without_ngraph_bn_op(
    const float epsilon, const NgraphNodePtr ng_maybe_gamma,
    const NgraphNodePtr ng_beta, const NgraphNodePtr ng_in_data,
    const size_t channel_axis) {
  check(channel_axis < ng_in_data->get_shape().size());
  return create_normalization_subgraph(epsilon, ng_maybe_gamma, ng_beta,
                                       ng_in_data, {channel_axis});
}

NgraphNodePtr create_batchnorm_inference_without_ngraph_bn_op(
    const float epsilon, const NgraphNodePtr ng_maybe_gamma,
    const NgraphNodePtr ng_beta, const NgraphNodePtr ng_in_data,
    const NgraphNodePtr ng_moving_mean, const NgraphNodePtr ng_moving_var,
    const size_t channel_axis) {
  const ngraph::Shape& batch_data_shape = ng_in_data->get_shape();
  const size_t batch_data_rank = batch_data_shape.size();

  check(channel_axis < ng_in_data->get_shape().size());

  return create_normalization_subgraph(epsilon, ng_maybe_gamma, ng_beta,
                                       ng_in_data, ng_moving_mean,
                                       ng_moving_var, {channel_axis});
}

ngraph::AxisVector channel_to_inverted_axes(const size_t rank,
                                            const size_t channel_axis) {
  ngraph::AxisVector axes;
  for (size_t i = 0; i < rank; ++i) {
    if (i != channel_axis) {
      axes.push_back(i);
    }
  }
  std::cout << "channel_axes " << axes << std::endl;
  return axes;
}

std::tuple<NgraphNodePtr, NgraphNodePtr, NgraphNodePtr>
create_layernorm_training_without_ngraph_bn_op(
    const float epsilon, const NgraphNodePtr ng_maybe_gamma,
    const NgraphNodePtr ng_beta, const NgraphNodePtr ng_in_data,
    const size_t channel_axis) {
  std::cout << "channel_axis " << channel_axis << std::endl;
  return create_normalization_subgraph(
      epsilon, ng_maybe_gamma, ng_beta, ng_in_data,
      channel_to_inverted_axes(ng_in_data->get_shape().size(), channel_axis),
      false);
}

NgraphNodePtr create_layernorm_inference_without_ngraph_bn_op(
    const float epsilon, const NgraphNodePtr ng_maybe_gamma,
    const NgraphNodePtr ng_beta, const NgraphNodePtr ng_in_data,
    const NgraphNodePtr ng_moving_mean, const NgraphNodePtr ng_moving_var,
    const size_t channel_axis) {
  return create_normalization_subgraph(
      epsilon, ng_maybe_gamma, ng_beta, ng_in_data, ng_moving_mean,
      ng_moving_var,
      channel_to_inverted_axes(ng_in_data->get_shape().size(), channel_axis),
      false);
}

}  // namespace ngraph_bridge
