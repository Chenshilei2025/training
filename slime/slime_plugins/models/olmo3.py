from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec


def get_olmo3_spec(args, config, vp_stage):
    transformer_layer_spec = get_gpt_layer_local_spec(
        args.num_experts,
        args.moe_grouped_gemm,
        args.qk_layernorm,
        args.multi_latent_attention,
        moe_use_legacy_grouped_gemm=getattr(args, "moe_use_legacy_grouped_gemm", False),
        normalization=args.normalization,
        post_self_attn_layernorm=True,
        post_mlp_layernorm=True,
    )
    return transformer_layer_spec
