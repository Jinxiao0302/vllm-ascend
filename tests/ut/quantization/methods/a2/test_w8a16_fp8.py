from unittest.mock import MagicMock, Mock, patch

import torch

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import (
    create_linear_layer,
    create_mock_ascend_config,
    create_mock_vllm_config,
    create_moe_layer,
)
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.quantization.methods.w8a16_fp8 import (
    AscendW8A16FP8FusedMoEMethod,
    AscendW8A16FP8LinearMethod,
)


class TestAscendW8A16FP8LinearMethod(TestBase):
    def setUp(self):
        self.method = AscendW8A16FP8LinearMethod()

    def test_get_weight_various_sizes(self):
        sizes = [(64, 128), (256, 512), (1024, 2048)]
        for input_size, output_size in sizes:
            weight = self.method.get_weight(input_size, output_size, torch.bfloat16)
            self.assertEqual(weight["weight"].dtype, torch.float8_e4m3fn)
            self.assertEqual(weight["weight"].shape, (output_size, input_size))

    def test_get_weight_preserves_fp8_dtype(self):
        weight = self.method.get_weight(256, 128, torch.float16)
        self.assertEqual(weight["weight"].dtype, torch.float8_e4m3fn)

    def test_get_perchannel_param(self):
        for output_size, dtype in [(128, torch.bfloat16), (256, torch.float16)]:
            params = self.method.get_perchannel_param(output_size, dtype)
            self.assertEqual(params["weight_scale"].dtype, torch.float32)
            self.assertEqual(params["weight_scale"].shape, (output_size, 1))
            self.assertEqual(len(params), 1)

    @patch("torch_npu.npu_quant_matmul")
    @patch("torch_npu.npu_dynamic_quant")
    def test_apply_2d_input(self, mock_dyn_quant, mock_matmul):
        mock_dyn_quant.return_value = (
            torch.randn(32, 128, dtype=torch.float8_e4m3fn),
            torch.randn(32, dtype=torch.float32),
        )
        mock_matmul.return_value = torch.randn(32, 256)
        layer = MagicMock()
        layer.weight = torch.randn(128, 256, dtype=torch.float8_e4m3fn)
        layer.weight_scale = torch.randn(256, dtype=torch.float32)
        x = torch.randn(32, 128, dtype=torch.bfloat16)
        output = self.method.apply(layer, x)
        mock_dyn_quant.assert_called_once()
        mock_matmul.assert_called_once()
        self.assertEqual(output.shape, (32, 256))

    @patch("torch_npu.npu_quant_matmul")
    @patch("torch_npu.npu_dynamic_quant")
    def test_apply_3d_input_with_squeeze(self, mock_dyn_quant, mock_matmul):
        mock_dyn_quant.return_value = (
            torch.randn(32, 1, 128, dtype=torch.float8_e4m3fn),
            torch.randn(32, 1, dtype=torch.float32),
        )
        mock_matmul.return_value = torch.randn(32, 1, 256)
        layer = MagicMock()
        layer.weight = torch.randn(128, 256, dtype=torch.float8_e4m3fn)
        layer.weight_scale = torch.randn(256, dtype=torch.float32)
        x = torch.randn(32, 1, 128, dtype=torch.bfloat16)
        output = self.method.apply(layer, x)
        mock_dyn_quant.assert_called_once()
        mock_matmul.assert_called_once()
        self.assertEqual(output.shape, (32, 1, 1, 256))

    @patch("torch_npu.npu_quant_matmul")
    @patch("torch_npu.npu_dynamic_quant")
    def test_apply_with_bias(self, mock_dyn_quant, mock_matmul):
        mock_dyn_quant.return_value = (
            torch.randn(32, 128, dtype=torch.float8_e4m3fn),
            torch.randn(32, dtype=torch.float32),
        )
        mock_matmul.return_value = torch.randn(32, 256)
        layer = MagicMock()
        layer.weight = torch.randn(128, 256, dtype=torch.float8_e4m3fn)
        layer.weight_scale = torch.randn(256, dtype=torch.float32)
        x = torch.randn(32, 128, dtype=torch.bfloat16)
        bias = torch.randn(256, dtype=torch.bfloat16)
        output = self.method.apply(layer, x, bias)
        mock_matmul.assert_called_once()
        self.assertEqual(output.shape, (32, 256))

    def test_process_weights_after_loading(self):
        layer = MagicMock()
        layer.weight.data = torch.randn(128, 256, dtype=torch.float8_e4m3fn)
        layer.weight_scale.data = torch.randn(256, 1, dtype=torch.float32)
        self.method.process_weights_after_loading(layer)
        self.assertEqual(layer.weight.data.shape, (256, 128))
        self.assertEqual(layer.weight_scale.data.shape, (256,))
        self.assertEqual(layer.weight_scale_fp32.dtype, torch.float32)
        self.assertEqual(layer.weight_scale_fp32.shape, (256,))


class TestAscendW8A16FP8LinearMethodWithNpu(TestBase):
    def setUp(self):
        self.method = AscendW8A16FP8LinearMethod()
        self.mock_get_config = patch("vllm_ascend.utils.get_ascend_config")
        mock_config = self.mock_get_config.start()
        mock_ascend_config = MagicMock()
        mock_ascend_config.weight_nz_mode = 0
        mock_config.return_value = mock_ascend_config

    def tearDown(self):
        self.mock_get_config.stop()

    def test_apply_with_npu(self):
        input_size, output_size = 128, 256
        params_dtype = torch.bfloat16
        layer = create_linear_layer(self.method, input_size, output_size, params_dtype)
        self.method.process_weights_after_loading(layer)

        x = torch.randn(32, input_size, dtype=params_dtype).npu()
        bias = torch.randn(output_size, dtype=torch.float32).npu()

        output = self.method.apply(layer, x, bias)
        self.assertEqual(output.shape, (32, output_size))


class TestAscendW8A16FP8FusedMoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 128

    @patch("vllm_ascend.quantization.methods.w8a16_fp8.get_ascend_config")
    def setUp(self, mock_ascend):
        with patch("vllm_ascend.quantization.methods.w8a16_fp8.get_current_vllm_config") as mock_vllm:
            mock_vllm.return_value = create_mock_vllm_config()
            mock_ascend.return_value = create_mock_ascend_config()
            self.quant_method = AscendW8A16FP8FusedMoEMethod()

    def test_quant_type_is_w8a16fp8(self):
        from vllm_ascend.quantization.quant_type import QuantType

        self.assertEqual(self.quant_method.quant_type, QuantType.W8A16FP8)

    def test_get_weight_dtype_is_float8_e4m3fn(self):
        param_dict = self.quant_method.get_weight(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
        )
        self.assertEqual(param_dict["w13_weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(param_dict["w2_weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(
            param_dict["w13_weight"].shape,
            (self.num_experts, 2 * self.intermediate_size, self.hidden_size),
        )
        self.assertEqual(
            param_dict["w2_weight"].shape,
            (self.num_experts, self.hidden_size, self.intermediate_size),
        )

    def test_get_weight_various_expert_counts(self):
        expert_counts = [4, 8, 16, 32]
        for num_experts in expert_counts:
            param_dict = self.quant_method.get_weight(
                num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
            )
            self.assertEqual(param_dict["w13_weight"].shape[0], num_experts)
            self.assertEqual(param_dict["w2_weight"].shape[0], num_experts)

    def test_get_dynamic_quant_param_scale_dtype(self):
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
        )
        self.assertEqual(param_dict["w13_weight_scale"].dtype, torch.float32)
        self.assertEqual(param_dict["w2_weight_scale"].dtype, torch.float32)
        self.assertEqual(
            param_dict["w13_weight_scale"].shape,
            (self.num_experts, 2 * self.intermediate_size, 1),
        )
        self.assertEqual(
            param_dict["w2_weight_scale"].shape,
            (self.num_experts, self.hidden_size, 1),
        )

    def test_get_dynamic_quant_param_offset_shape(self):
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
        )
        self.assertEqual(
            param_dict["w13_weight_offset"].shape,
            (self.num_experts, 2 * self.intermediate_size, 1),
        )
        self.assertEqual(
            param_dict["w2_weight_offset"].shape,
            (self.num_experts, self.hidden_size, 1),
        )
        self.assertTrue(torch.all(param_dict["w13_weight_offset"] == 0))
        self.assertTrue(torch.all(param_dict["w2_weight_offset"] == 0))

    @patch("vllm_ascend.quantization.methods.w8a16_fp8._EXTRA_CTX")
    @patch("vllm_ascend.quantization.methods.w8a16_fp8.select_experts")
    def test_apply_uses_explicit_dispatch_and_mlp_args(self, mock_select_experts, mock_extra_ctx):
        tokens = 4
        hidden_size = self.hidden_size
        layer = torch.nn.Module()
        layer.w13_weight = torch.randn(
            self.num_experts, 2 * self.intermediate_size, hidden_size, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        layer.w2_weight = torch.randn(
            self.num_experts, hidden_size, self.intermediate_size, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        layer.w13_weight_scale_fp32 = torch.ones(
            self.num_experts, 2 * self.intermediate_size, dtype=torch.float32
        )
        layer.w2_weight_scale = torch.ones(self.num_experts, hidden_size, dtype=torch.float32)
        layer.w13_weight_offset = torch.zeros(self.num_experts, 2 * self.intermediate_size)
        layer.w2_weight_offset = torch.zeros(self.num_experts, hidden_size)
        layer.swiglu_limit = 1000000

        x = torch.randn(tokens, hidden_size, dtype=torch.float32)
        router_logits = torch.randn(tokens, self.num_experts, dtype=torch.float32)
        topk_weights = torch.randn(tokens, 2, dtype=torch.float32)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2), dtype=torch.int64)
        mc2_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        pertoken_scale = torch.randn(tokens, dtype=torch.float32)

        mock_select_experts.return_value = (topk_weights, topk_ids)
        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.randn(tokens, hidden_size, dtype=torch.float32)
        mock_extra_ctx.moe_comm_method = mock_comm
        mock_extra_ctx.moe_comm_type = MoECommType.ALLGATHER
        self.quant_method.multistream_overlap_gate = False
        self.quant_method.in_dtype = torch.float32

        self.quant_method.apply(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=2,
            renormalize=True,
            num_experts=self.num_experts,
            activation="gelu",
            apply_router_weight_on_input=True,
            mc2_mask=mc2_mask,
            pertoken_scale=pertoken_scale,
        )

        fused_experts_input = mock_comm.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertEqual(fused_experts_input.activation, "gelu")
        self.assertTrue(fused_experts_input.routing.apply_router_weight_on_input)
        self.assertIs(fused_experts_input.routing.mc2_mask, mc2_mask)
        self.assertIs(fused_experts_input.routing.pertoken_scale, pertoken_scale)
        self.assertIs(fused_experts_input.topk_weights, topk_weights)
        self.assertIs(fused_experts_input.topk_ids, topk_ids)

    @patch("vllm_ascend.quantization.methods.w8a16_fp8._EXTRA_CTX")
    @patch("vllm_ascend.quantization.methods.w8a16_fp8.select_experts")
    def test_apply_with_zero_experts(self, mock_select_experts, mock_extra_ctx):
        tokens = 4
        hidden_size = self.hidden_size
        layer = torch.nn.Module()
        layer.zero_expert_num = 2
        layer.zero_expert_type = "mean"
        layer.w13_weight = torch.randn(
            self.num_experts, 2 * self.intermediate_size, hidden_size, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        layer.w2_weight = torch.randn(
            self.num_experts, hidden_size, self.intermediate_size, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        layer.w13_weight_scale_fp32 = torch.ones(
            self.num_experts, 2 * self.intermediate_size, dtype=torch.float32
        )
        layer.w2_weight_scale = torch.ones(self.num_experts, hidden_size, dtype=torch.float32)
        layer.w13_weight_offset = torch.zeros(self.num_experts, 2 * self.intermediate_size)
        layer.w2_weight_offset = torch.zeros(self.num_experts, hidden_size)
        layer.swiglu_limit = 1000000

        x = torch.randn(tokens, hidden_size, dtype=torch.float32)
        router_logits = torch.randn(tokens, self.num_experts, dtype=torch.float32)
        topk_weights = torch.randn(tokens, 2, dtype=torch.float32)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2), dtype=torch.int64)

        mock_select_experts.return_value = (topk_weights, topk_ids)
        mock_comm = Mock()
        mock_zero_result = torch.randn(tokens, hidden_size, dtype=torch.float32)
        mock_comm.fused_experts.return_value = mock_zero_result
        mock_extra_ctx.moe_comm_method = mock_comm
        mock_extra_ctx.moe_comm_type = MoECommType.ALLGATHER
        self.quant_method.multistream_overlap_gate = False
        self.quant_method.in_dtype = torch.float32

        with patch(
            "vllm_ascend.quantization.methods.w8a16_fp8.zero_experts_compute"
        ) as mock_zero:
            mock_zero.return_value = (topk_ids, topk_weights, mock_zero_result)
            output = self.quant_method.apply(
                layer=layer,
                x=x,
                router_logits=router_logits,
                top_k=2,
                renormalize=True,
                num_experts=self.num_experts,
            )
            self.assertEqual(output.shape, (tokens, hidden_size))

    @patch("vllm_ascend.quantization.methods.w8a16_fp8.get_ascend_config")
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_fused_mc2(self, mock_format_cast, mock_get_config):
        mock_config = MagicMock()
        mock_config.enable_fused_mc2 = 1
        mock_get_config.return_value = mock_config
        mock_format_cast.return_value = torch.randn(
            self.num_experts, self.hidden_size, 2 * self.intermediate_size, dtype=torch.float8_e4m3fn
        )
        self.quant_method.dynamic_eplb = True
        layer = create_moe_layer(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            weight_dtype=torch.float8_e4m3fn,
            params_dtype=torch.float32,
        )
        self.quant_method.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "w13_weight_list"))
        self.assertTrue(hasattr(layer, "w2_weight_list"))
        self.assertTrue(hasattr(layer, "fused_w1_scale_list"))
        self.assertTrue(hasattr(layer, "fused_w2_scale_list"))
        self.assertFalse(hasattr(layer, "w13_weight_scale_fp32"))

    @patch("vllm_ascend.quantization.methods.w8a16_fp8.get_ascend_config")
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_without_dynamic_eplb(self, mock_format_cast, mock_get_config):
        mock_config = MagicMock()
        mock_config.enable_fused_mc2 = 0
        mock_get_config.return_value = mock_config
        mock_format_cast.return_value = torch.randn(
            self.num_experts, self.hidden_size, 2 * self.intermediate_size, dtype=torch.float8_e4m3fn
        )
        self.quant_method.dynamic_eplb = False
        layer = create_moe_layer(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            weight_dtype=torch.float8_e4m3fn,
            params_dtype=torch.float32,
        )
        self.quant_method.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "w13_weight"))
        self.assertTrue(hasattr(layer, "w2_weight"))
        self.assertTrue(hasattr(layer, "w13_weight_scale_fp32"))
        self.assertTrue(hasattr(layer, "w2_weight_scale_fp32"))
        self.assertEqual(layer.w13_weight_scale_fp32.dtype, torch.float32)
        self.assertEqual(layer.w2_weight_scale_fp32.dtype, torch.float32)