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

#ifndef MXNET_KVSTORE_KVSTORE_NGRAPH_H_
#define MXNET_KVSTORE_KVSTORE_NGRAPH_H_

#if MXNET_USE_NGRAPH_DISTRIBUTED

#include <mxnet/kvstore.h>
#include <ngraph/distributed.hpp>
#include "../../src/kvstore/kvstore_local.h"

namespace ngraph_bridge {

/**
 * \brief store data in local machine using nGraph
 */
class KVStoreNGRAPH : public mxnet::kvstore::KVStoreLocal {
 public:
  explicit KVStoreNGRAPH(bool use_device_comm)
      : mxnet::kvstore::KVStoreLocal(use_device_comm) {
    dmlc::SetEnv("MXNET_NGRAPH_DISTRIBUTED", 1);
  }

  virtual ~KVStoreNGRAPH() {}

  int get_group_size() const override { return dist.get_size(); }

  int get_rank() const override { return dist.get_rank(); }

 private:
  ngraph::Distributed dist;
};
}  // namespace ngraph_bridge
#endif
#endif
