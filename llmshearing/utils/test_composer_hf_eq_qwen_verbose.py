import torch
from omegaconf import OmegaConf as om
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmshearing.models.composer_qwen import ComposerMosaicQwen

def construct_example_cfg(model_size, path=None, add_l0_module=False):
    """ construct example cfg for Qwen2 models """
    if model_size == "0.5B":
        cfg = om.create({
            "name": "qwen2_0.5b",
            "init_device": "cpu",
            "d_model": 896,
            "n_heads": 14,
            "n_kv_heads": 2,
            "n_layers": 24,
            "intermediate_size": 4864,
            "max_seq_len": 32768,
            "vocab_size": 151936,
            "init_std": 0.02,
            "attn_pdrop": 0.0,
            "resid_pdrop": 0.0,
            "emb_pdrop": 0.0,
            "attn_impl": "flash",
            "rms_norm_eps": 1e-6,
            "model_type": "qwen2",
            "bos_token_id": 151643,
            "eos_token_id": 151643,
            "hidden_act": "silu",
            "use_cache": True,
            "use_sliding_window": False,
            "sliding_window": 32768,
            "rope_theta": 1000000.0,
            "use_mrope": False,
            "path": path
        })
    
    if add_l0_module:
        cfg["l0_module"] = {
            "start_sparsity": 0,
            "target_sparsity": 0.6,
            "pruning_modules": ["head", "head_layer", "mlp", "intermediate", "hidden"],
            "lagrangian_warmup_steps": "320ba"
        }
    return cfg

def test_two_matrix(a, b, desc=""):
    """ test if two matrix are equal """
    s1 = a.sum().item()
    s2 = b.sum().item() if b is not None else torch.tensor(0).to(a.device).to(a.dtype)
    try:
        assert abs(s1 - s2) < 1e-3
    except:
        print(f"[{desc}] failed! sums are not equal: {s1} vs {s2}")
        return 
    print(f"[{desc}] passed! sums are equal: {s1} vs {s2}")

def compare_model_outputs(hf_model, composer_model, input_ids):
    """Compare intermediate outputs between HF and Composer models"""
    
    # Get HF outputs
    hf_outputs = hf_model(input_ids, output_hidden_states=True)
    hf_hidden_states = hf_outputs.hidden_states
    
    # Get Composer outputs with hooks
    composer_hidden_states = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            composer_hidden_states.append(output[0].detach())
        else:
            composer_hidden_states.append(output.detach())
    
    # Register hooks for each layer
    hooks = []
    for block in composer_model.model.transformer.blocks:
        hooks.append(block.register_forward_hook(hook_fn))
    
    composer_outputs = composer_model({"input_ids": input_ids})
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compare layer by layer
    print("\nLayer-wise comparison:")
    for i, (hf_hidden, composer_hidden) in enumerate(zip(hf_hidden_states[1:], composer_hidden_states)):
        test_two_matrix(hf_hidden.mean(), composer_hidden.mean(), f"Layer {i}")
    
    # Compare final logits
    print("\nFinal logits comparison:")
    test_two_matrix(hf_outputs.logits.mean(), composer_outputs["logits"].mean(), "Final logits")

if __name__ == "__main__":
    import sys
    
    hf_qwen_path = sys.argv[1]
    composer_qwen_path = sys.argv[2]
    model_size = sys.argv[3]
    
    tokenizer = AutoTokenizer.from_pretrained(hf_qwen_path)
    text = "Chamath Palihapitiya (born 3 September 1976)[1] is a Sri Lankan-born Canadian and American venture capitalist, engineer, SPAC sponsor, founder and CEO of Social Capital. Palihapitiya was an early senior executive at Facebook, working at the company from 2007 to 2011. Following his departure from Facebook, Palihapitiya started his fund, The Social+Capital Partnership, through which he invested in several companies, including Yammer and Slack. "
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Load models
    hf_model = AutoModelForCausalLM.from_pretrained(hf_qwen_path)
    cfg = construct_example_cfg(model_size)
    composer_model = ComposerMosaicQwen(cfg)
    composer_model.load_state_dict(torch.load(composer_qwen_path), strict=False)

    # Move to GPU and convert to bfloat16
    input_ids = input_ids.cuda()
    composer_model.bfloat16().cuda()
    hf_model.bfloat16().cuda()

    # Compare outputs
    compare_model_outputs(hf_model, composer_model, input_ids)