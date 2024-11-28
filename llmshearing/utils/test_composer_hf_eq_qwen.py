import torch
from omegaconf import OmegaConf as om
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmshearing.models.composer_qwen import ComposerMosaicQwen


def construct_example_cfg(model_size, path=None, add_l0_module=False):
    """construct example cfg for qwen models"""
    if model_size == "0.5B":
        cfg = om.create(
            {
                "name": "qwen-0.5b",
                "d_model": 896,
                "n_heads": 14,
                "n_layers": 24,
                "n_kv_heads": 2,
                "intermediate_size": 4864,
                "rms_norm_eps": 1e-6,
            }
        )
    else:
        raise ValueError(f"model size {model_size} not supported")

    # add default values
    cfg = om.merge(
        cfg,
        om.create(
            {
                "init_device": "cpu",
                "init_std": 0.02,
                "attn_impl": "flash",
                "rotary_emb_base": 1000000,
                "vocab_size": 151936,
                "max_seq_len": 32768,
            }
        ),
    )

    if add_l0_module:
        cfg["l0_module"] = {
            "start_sparsity": 0,
            "target_sparsity": 0.6,
            "pruning_modules": ["head", "head_layer", "mlp", "intermediate", "hidden"],
            "lagrangian_warmup_steps": "320ba",
        }
    return cfg


def test_two_matrix(a, b, desc=""):
    """test if two matrix are equal"""
    s1 = a.sum().item()
    s2 = b.sum().item() if b is not None else torch.tensor(0).to(a.device).to(a.dtype)
    try:
        assert abs(s1 - s2) < 1e-3
    except:
        print(f"[{desc}] failed! sums are not equal: {s1} vs {s2}")
        return
    print(f"[{desc}] passed! sums are equal: {s1} vs {s2}")


if __name__ == "__main__":
    import sys

    hf_qwen_path = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
    composer_qwen_path = (
        sys.argv[2] if len(sys.argv) > 2 else "models/qwen-0.5b-composer/state_dict.pt"
    )
    model_size = sys.argv[3] if len(sys.argv) > 3 else "0.5B"

    cfg = construct_example_cfg(model_size)
    composer_model = ComposerMosaicQwen(cfg)
    # Disable strict mode to allow for missing rotary_emb.inv_freq
    composer_model.load_state_dict(torch.load(composer_qwen_path), strict=False)

    print(composer_model.state_dict().keys())

    # check if they have the same naming convention
    hf_model = AutoModelForCausalLM.from_pretrained(hf_qwen_path)

    tokenizer = AutoTokenizer.from_pretrained(hf_qwen_path)
    text = "Chamath Palihapitiya (born 3 September 1976)[1] is a Sri Lankan-born Canadian and American venture capitalist, engineer, SPAC sponsor, founder and CEO of Social Capital. Palihapitiya was an early senior executive at Facebook, working at the company from 2007 to 2011. Following his departure from Facebook, Palihapitiya started his fund, The Social+Capital Partnership, through which he invested in several companies, including Yammer and Sklack. "
    input_ids = tokenizer.encode(text, return_tensors="pt")

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        composer_model.half().cuda()
        hf_model.half().cuda()

    hf_output = hf_model(input_ids, labels=input_ids)
    composer_output = composer_model({"input_ids": input_ids})

    logits1 = hf_output.logits.mean()
    logits2 = composer_output["logits"].mean()

    test_two_matrix(logits1, logits2, "HF vs. Composer")
