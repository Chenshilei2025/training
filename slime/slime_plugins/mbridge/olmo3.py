"""Megatron/HF bridge for OLMo-3 dense models."""

import torch

from mbridge.core import register_model
from mbridge.models import Qwen2Bridge


@register_model("olmo3")
class OLMo3Bridge(Qwen2Bridge):
    _OTHER_MAPPING = {
        "post_attention_layernorm.weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "post_feedforward_layernorm.weight": ["model.layers.{layer_number}.post_feedforward_layernorm.weight"],
    }

    def _build_config(self):
        return self._build_base_config(
            use_cpu_initialization=False,
            qk_layernorm=True,
            add_qkv_bias=False,
            add_bias_linear=False,
            rotary_interleaved=False,
        )

    def _weight_name_mapping_mcore_to_hf(self, name):
        if "pre_mlp_layernorm.weight" in name or "input_layernorm.weight" in name:
            layer = name.split(".")[2]
            return [f"model.layers.{layer}.post_attention_layernorm.weight"]
        return super()._weight_name_mapping_mcore_to_hf(name)

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        if mcore_weights_name.endswith(
            ("self_attention.q_layernorm.weight", "self_attention.k_layernorm.weight")
        ):
            assert len(hf_weights) == 1
            weight = hf_weights[0]
            hidden_size = self.hf_config.hidden_size
            num_attention_heads = self.hf_config.num_attention_heads
            head_dim = getattr(self.hf_config, "head_dim", hidden_size // num_attention_heads)

            if weight.numel() == head_dim:
                return weight.contiguous()
            if weight.numel() != hidden_size:
                raise ValueError(
                    f"Unexpected OLMo3 q/k norm shape for {mcore_weights_name}: "
                    f"{tuple(weight.shape)}; expected {head_dim} or {hidden_size} elements"
                )

            # Megatron local q/k norms are head_dim-sized and shared across
            # local heads, while HF OLMo3 stores one scale per hidden channel.
            return weight.view(num_attention_heads, head_dim).float().mean(dim=0).to(weight.dtype).contiguous()

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        hf_names, hf_weights = super()._weight_to_hf_format(mcore_weights_name, mcore_weights)
        if mcore_weights_name.endswith("self_attention.q_layernorm.weight") and len(hf_weights) == 1:
            return hf_names, [hf_weights[0].repeat(self.hf_config.num_attention_heads).contiguous()]
        if mcore_weights_name.endswith("self_attention.k_layernorm.weight") and len(hf_weights) == 1:
            return hf_names, [hf_weights[0].repeat(self.hf_config.num_key_value_heads).contiguous()]
        return hf_names, hf_weights
