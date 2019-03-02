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

#include "test_ngraph_interface.h"
#include "ngraph_nnvm_ops.h"
#include "ngraph_sgcompiler.h"
#include <vector>

namespace ngraph_bridge {

TEST_F(NGRAPH_INTERFACE, NULLNODE_SUBGRAPH_SELECTOR) {
  nodes_["add1"]->inputs[0].node = nullptr;
  EXPECT_ANY_THROW(subgraph_prop->CreateSubgraphSelector());
}

TEST_F(NGRAPH_INTERFACE, NULLNODE_SUBGRAPH_NODE) {
  nodes_["add1"]->inputs[0].node = nullptr;
  EXPECT_ANY_THROW(subgraph_prop->CreateSubgraphNode(nnvm_graph));
}

TEST_F(NGRAPH_INTERFACE, INFER_BAD_SHAPES) {
  std::vector<mxnet::TShape> small(3, mxnet::TShape{2,2});
  std::vector<mxnet::TShape> good(4, mxnet::TShape{2,2});

  static auto& finfer_generator =
      mxnet::Op::GetAttr<nnvm::FInferShape>("FInferShape");
  auto finfer = finfer_generator.get(compiled_subgraph->op(), nullptr);

  EXPECT_ANY_THROW(finfer(compiled_subgraph->attrs, &small, &good));
  EXPECT_ANY_THROW(finfer(compiled_subgraph->attrs, &good, &small));
  EXPECT_ANY_THROW(finfer(compiled_subgraph->attrs, nullptr, &good));
  EXPECT_ANY_THROW(finfer(compiled_subgraph->attrs, &good, nullptr));
}

TEST_F(NGRAPH_INTERFACE, INFER_BAD_TYPES) {
  std::vector<int> small(3, 0);
  std::vector<int> good(4, 0);

  static auto& finfer_generator =
      mxnet::Op::GetAttr<nnvm::FInferType>("FInferType");
  auto finfer = finfer_generator.get(compiled_subgraph->op(), nullptr);

  EXPECT_ANY_THROW(finfer(compiled_subgraph->attrs, &small, &good));
  EXPECT_ANY_THROW(finfer(compiled_subgraph->attrs, &good, &small));
  EXPECT_ANY_THROW(finfer(compiled_subgraph->attrs, nullptr, &good));
  EXPECT_ANY_THROW(finfer(compiled_subgraph->attrs, &good, nullptr));
}

TEST_F(NGRAPH_INTERFACE, INFER_NULL_STORAGE) {
  std::vector<int> small(3, 0);
  std::vector<int> good(4, 0);
  mxnet::DispatchMode dispatch_mode(mxnet::DispatchMode::kFComputeEx);

  static auto& finfer_generator =
      mxnet::Op::GetAttr<mxnet::FInferStorageType>("FInferStorageType");
  auto finfer = finfer_generator.get(compiled_subgraph->op(), nullptr);

  EXPECT_ANY_THROW(finfer(compiled_subgraph->attrs, 0, nullptr, &small, &good));
  EXPECT_ANY_THROW(
      finfer(compiled_subgraph->attrs, 0, &dispatch_mode, nullptr, &good));
  EXPECT_ANY_THROW(
      finfer(compiled_subgraph->attrs, 0, &dispatch_mode, &small, nullptr));
}

TEST_F(NGRAPH_INTERFACE, MISSING_DATA_COMPUTE_FORWARD) {
  mxnet::TShape shape{2, 2};
  std::vector<float> vec1{1, 1, 1, 1};

  std::vector<mxnet::NDArray> one(
      1, mxnet::NDArray(mxnet::TBlob(vec1.data(), shape, 1, 0), 0));
  std::vector<mxnet::NDArray> two(
      2, mxnet::NDArray(mxnet::TBlob(vec1.data(), shape, 1, 0), 0));

  std::vector<mxnet::OpReqType> one_req(1, mxnet::OpReqType::kWriteTo);
  std::vector<mxnet::OpReqType> two_req(2, mxnet::OpReqType::kWriteTo);

  auto compiler = nnvm::get<std::shared_ptr<ngraph_bridge::Compiler>>(
      compiled_subgraph->attrs.parsed);
  auto graph = compiler->GetNgraph();
  mxnet::OpContext ctx;
  ctx.is_train = false;
  EXPECT_ANY_THROW(
      ngraph_bridge::compute_forward(ctx, graph, one, one_req, one));
  EXPECT_ANY_THROW(
      ngraph_bridge::compute_forward(ctx, graph, two, two_req, {}));
  EXPECT_ANY_THROW(
      ngraph_bridge::compute_forward(ctx, graph, two, one_req, two));
}

TEST_F(NGRAPH_INTERFACE, MISSING_DATA_COMPUTE_BACKWARD) {
  auto compiler = nnvm::get<std::shared_ptr<ngraph_bridge::Compiler>>(
      compiled_subgraph->attrs.parsed);
  auto graph = compiler->GetNgraph();
  if (graph->ngraph_backward[1] == nullptr) {
    CompileForwardBackward(graph, graph->fprop_cache->fprop,
                           graph->fprop_cache->bprop, GraphExeMode::kTrain,
                           *(graph->fprop_cache));
  }

  auto num_inputs = graph->ngraph_backward[1]->get_parameters().size();
  auto num_outputs = graph->ngraph_backward[1]->get_results().size();

  mxnet::TShape shape{2, 2};
  std::vector<float> vec1{1, 1, 1, 1};

  std::vector<mxnet::NDArray> full_input(
      num_inputs, mxnet::NDArray(mxnet::TBlob(vec1.data(), shape, 1, 0), 0));
  std::vector<mxnet::NDArray> small_input(
      num_inputs - 1,
      mxnet::NDArray(mxnet::TBlob(vec1.data(), shape, 1, 0), 0));
  std::vector<mxnet::NDArray> full_output(
      num_outputs, mxnet::NDArray(mxnet::TBlob(vec1.data(), shape, 1, 0), 0));
  std::vector<mxnet::NDArray> small_output(
      num_outputs - 1,
      mxnet::NDArray(mxnet::TBlob(vec1.data(), shape, 1, 0), 0));

  std::vector<mxnet::OpReqType> small_req(num_outputs - 1,
                                          mxnet::OpReqType::kWriteTo);
  std::vector<mxnet::OpReqType> full_req(num_outputs,
                                         mxnet::OpReqType::kWriteTo);
  mxnet::OpContext ctx;
  ctx.is_train = true;
  EXPECT_ANY_THROW(ngraph_bridge::compute_backward(ctx, graph, small_input,
                                                   full_req, full_output));
  EXPECT_ANY_THROW(ngraph_bridge::compute_backward(ctx, graph, full_input,
                                                   small_req, full_output));
  EXPECT_ANY_THROW(ngraph_bridge::compute_backward(ctx, graph, full_input,
                                                   small_req, small_output));
}

}  // namespace ngraph_bridge
