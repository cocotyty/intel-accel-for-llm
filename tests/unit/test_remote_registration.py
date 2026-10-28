# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run in the dev container: python3 -m unittest discover -s tests/unit -v."""

import unittest
from unittest.mock import Mock, patch

import torch

from iaxl.remote_pool import kvstore_remote, rpc


class RemoteRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.xfer = Mock()
        self.channel = Mock()
        self.regions = set()
        self.xfer.register_memory.side_effect = self.register_memory
        self.xfer.connect.side_effect = self.connect
        for target, value in (
            ("rdma_xfer", Mock(return_value=self.xfer)),
            ("RpcChannel", Mock(return_value=self.channel)),
        ):
            patcher = patch.object(kvstore_remote, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(kvstore_remote.envs, "IAXL_RDMA_CLIENT_NIC_IPS", "127.0.0.1")
        patcher.start()
        self.addCleanup(patcher.stop)

    def register_memory(self, tensor):
        # A remote metadata importer must never see a region twice.
        region = (tensor.data_ptr(), tensor.numel() * tensor.element_size())
        self.assertNotIn(region, self.regions)
        self.regions.add(region)

    def connect(self, *args, **kwargs):
        self.regions_at_connect = self.regions.copy()

    def make_store(self, **kwargs):
        return kvstore_remote.KVStoreRemote("test-model", daemon_ip="127.0.0.1", **kwargs)

    def test_shared_views_register_once_but_keep_every_logical_layer(self):
        first = torch.zeros(4, 16)
        second = torch.zeros_like(first)
        caches = {
            "mamba.0": first.view_as(first),
            "attention.0": first.view_as(first),
            "mamba.1": second.view_as(second),
            "attention.1": second.view_as(second),
        }
        store = self.make_store(block_dim=0, kv_caches=caches)

        self.assertEqual(len(self.regions), 2)
        self.assertEqual(self.regions_at_connect, self.regions)
        method, payload = self.channel.call.call_args.args
        self.assertEqual(method, rpc.REGISTER_KV_CACHES)
        layers = rpc.unpack_json(payload)["layers"]
        self.assertEqual(list(layers), list(caches))
        self.assertEqual(store.layer_names, list(caches))
        self.assertEqual(store.layer_idx, {name: i for i, name in enumerate(caches)})
        for name, tensor in caches.items():
            self.assertEqual(layers[name]["base"], tensor.data_ptr())

    def test_distinct_regions_with_identical_geometry_are_all_registered(self):
        caches = {f"attention.{i}": torch.zeros(4, 16) for i in range(3)}
        self.make_store(block_dim=0, kv_caches=caches)
        self.assertEqual(len(self.regions), 3)
        self.assertEqual(self.regions_at_connect, self.regions)

    def test_scheduler_registers_layer_names_without_gpu_memory(self):
        names = ["mamba.0", "attention.0"]
        store = self.make_store(layer_names=names)
        self.xfer.register_memory.assert_not_called()
        method, payload = self.channel.call.call_args.args
        self.assertEqual(method, rpc.REGISTER_LAYERS)
        self.assertEqual(rpc.unpack_json(payload)["layer_names"], names)
        self.assertTrue(store.has_only_mode)


if __name__ == "__main__":
    unittest.main()
