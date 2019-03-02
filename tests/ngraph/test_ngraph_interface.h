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
#ifndef TESTS_CPP_NGRAPH_TEST_NGRAPH_INTERFACE_H_
#define TESTS_CPP_NGRAPH_TEST_NGRAPH_INTERFACE_H_

#include <nnvm/graph.h>

#include <string>

#include "ngraph_compiler.h"
#include "../src/operator/contrib/ngraph-inl.h"
#include "test_util.h"

namespace ngraph_bridge {

class NGRAPH_INTERFACE : public ::testing::Test {
 protected:
  nnvm::NodeEntry createNode(std::string name, std::string op = "") {
    nnvm::NodeAttrs attr;
    auto node = nnvm::Node::Create();
    attr.name = name;
    if (op != "") attr.op = nnvm::Op::Get(op);
    node->attrs = attr;
    nodes_[name] = node;
    return nnvm::NodeEntry{node, 0, 0};
  }

  virtual void SetUp() {
    auto A = createNode("A");
    auto B = createNode("B");
    auto add1 = createNode("add1", "_add");
    auto relu = createNode("relu", "relu");

    add1.node->inputs.push_back(A);
    add1.node->inputs.push_back(B);

    relu.node->inputs.push_back(add1);

    nnvm_graph.outputs.push_back(relu);

    mxnet::TShape shape{2, 2};

    mxnet::exec::ContextVector contexts(nodes_.size(), mxnet::Context::CPU());
    mxnet::ShapeVector shapes(nodes_.size(), shape);
    nnvm::DTypeVector types(nodes_.size(), 0);
    mxnet::StorageTypeVector stypes(nodes_.size(), mxnet::kDefaultStorage);
    mxnet::exec::DevMaskVector dev_masks(nodes_.size(), mxnet::Context::CPU().dev_mask());

    nnvm_graph.attrs["context"] = std::make_shared<dmlc::any>(std::move(contexts));

    nnvm_graph.attrs["shape"] = std::make_shared<dmlc::any>(std::move(shapes));
    nnvm_graph = mxnet::exec::InferShape(std::move(nnvm_graph));
    CHECK_EQ(nnvm_graph.GetAttr<size_t>("shape_num_unknown_nodes"), 0U);

    nnvm_graph.attrs["dtype"] = std::make_shared<dmlc::any>(std::move(types));
    nnvm_graph = mxnet::exec::InferType(std::move(nnvm_graph));
    CHECK_EQ(nnvm_graph.GetAttr<size_t>("dtype_num_unknown_nodes"), 0U);

    nnvm_graph.attrs["dev_mask"] = std::make_shared<dmlc::any>(std::move(dev_masks));

    nnvm_graph.attrs["storage_type"] = std::make_shared<dmlc::any>(std::move(stypes));
    nnvm_graph = mxnet::exec::InferStorageType(std::move(nnvm_graph));

    subgraph_prop =
        mxnet::op::SubgraphPropertyRegistry::Get()->CreateSubgraphProperty(
           "ngraph");
    
    subgraph_prop->SetAttr("graph", nnvm_graph);
    subgraph_prop->SetAttr(
        "grad_reqs", std::vector<mxnet::OpReqType>(nodes_.size(),
                                                   mxnet::OpReqType::kWriteTo));

    compiled_subgraph = subgraph_prop->CreateSubgraphNode(nnvm_graph);
  }

  virtual void TearDown() {}

  nnvm::Graph nnvm_graph;
  std::shared_ptr<nnvm::Node> compiled_subgraph;
  std::shared_ptr<ngraph_bridge::SimpleBindArg> bindarg;

  std::shared_ptr<mxnet::op::SubgraphProperty> subgraph_prop;

  NDArrayMap feed_dict;
  NNVMNodeVec inputs;
  std::unordered_map<std::string, nnvm::NodePtr> nodes_;
  std::unordered_map<std::string, int> dtypes;
  std::unordered_map<std::string, int> stypes;
  std::unordered_map<std::string, mxnet::TShape> shapes;
};

}  // namespace ngraph_bridge

#endif  // TESTS_CPP_NGRAPH_TEST_NGRAPH_COMPILER_H_
