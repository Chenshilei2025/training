from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec


def get_olmo3_spec(args, config, vp_stage):
    transformer_layer_spec = get_gpt_layer_local_spec(
        args.num_experts,
        args.moe_grouped_gemm,
        args.qk_layernorm,
        args.multi_latent_attention,
        normalization=args.normalization,
    )
    return transformer_layer_spec
