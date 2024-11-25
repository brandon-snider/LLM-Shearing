import torch
from omegaconf import OmegaConf as om
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmshearing.models.composer_pythia import ComposerMosaicPythia


def construct_example_cfg(model_size, path=None, add_l0_module=False):
    """construct example cfg for pythia models"""
    if model_size == "14m":
        cfg = om.create(
            {
                "name": "pythia-14m",
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 6,
                "intermediate_size": 512,
            }
        )
    elif model_size == "70m":
        cfg = om.create(
            {
                "name": "pythia-70m",
                "d_model": 512,
                "n_heads": 8,
                "n_layers": 6,
                "intermediate_size": 2048,
            }
        )
    elif model_size == "160m":
        cfg = om.create(
            {
                "name": "pythia-160m",
                "d_model": 768,
                "n_heads": 12,
                "n_layers": 12,
                "intermediate_size": 3072,
            }
        )
    elif model_size == "410m":
        cfg = om.create(
            {
                "name": "pythia-410m",
                "d_model": 1024,
                "n_heads": 16,
                "n_layers": 24,
                "intermediate_size": 4096,
            }
        )
    elif model_size == "410m-deduped":
        cfg = om.create(
            {
                "name": "pythia-410m-deduped",
                "d_model": 1024,
                "n_heads": 16,
                "n_layers": 24,
                "intermediate_size": 4096,
            }
        )
    elif model_size == "1.4b":
        cfg = om.create(
            {
                "name": "pythia-1.4b",
                "d_model": 2048,
                "n_heads": 16,
                "n_layers": 24,
                "intermediate_size": 8192,
            }
        )

    # add default values
    cfg = om.merge(
        cfg,
        om.create(
            {
                "init_device": "cpu",
                "init_std": 0.02,
                "rotary_pct": 0.25,
                "use_parallel_residual": True,
                "layer_norm_eps": 1e-5,
                "attn_pdrop": 0.0,
                "resid_pdrop": 0.0,
                "emb_pdrop": 0.0,
                "attn_impl": "flash",
                "rotary_emb_base": 10000,
                "vocab_size": 50304,
                "max_seq_len": 2048,
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

    hf_pythia_path = sys.argv[1]
    composer_pythia_path = sys.argv[2]
    model_size = sys.argv[3]

    tokenizer = AutoTokenizer.from_pretrained(hf_pythia_path)
    text = "Chamath Palihapitiya (born 3 September 1976)[1] is a Sri Lankan-born Canadian and American venture capitalist, engineer, SPAC sponsor, founder and CEO of Social Capital. Palihapitiya was an early senior executive at Facebook, working at the company from 2007 to 2011. Following his departure from Facebook, Palihapitiya started his fund, The Social+Capital Partnership, through which he invested in several companies, including Yammer and Sklack. "
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # check if they have the same naming convention
    hf_model = AutoModelForCausalLM.from_pretrained(hf_pythia_path)

    cfg = construct_example_cfg(model_size)
    composer_model = ComposerMosaicPythia(cfg)
    # rotary_emb.inv_freq can be missing
    composer_model.load_state_dict(torch.load(composer_pythia_path), strict=False)

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        composer_model.half().cuda()
        hf_model.half().cuda()

    hf_output = hf_model(input_ids, labels=input_ids)
    composer_output = composer_model({"input_ids": input_ids})

    logits1 = hf_output.logits.mean()
    logits2 = composer_output["logits"].mean()

    test_two_matrix(logits1, logits2, "HF vs. Composer")
