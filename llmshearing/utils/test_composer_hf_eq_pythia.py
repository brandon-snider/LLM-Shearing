import torch
from omegaconf import OmegaConf as om
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmshearing.models.composer_pythia import ComposerMosaicPythia


def construct_example_cfg(model_size, path=None, add_l0_module=False):
    """ construct example cfg for pythia models """
    if model_size == "410m":
        cfg = om.create({
            "name": "pythia-410m",
            "init_device": "cpu",
            "d_model": 1024,  # hidden_size in config
            "n_heads": 16,    # num_attention_heads
            "n_layers": 24,   # num_hidden_layers
            "intermediate_size": 4096,
            "max_seq_len": 2048,  # max_position_embeddings
            "vocab_size": 50304,
            "init_std": 0.02,     # initializer_range
            "rotary_pct": 0.25,
            "use_parallel_residual": True,
            "layer_norm_eps": 1e-5
        })
    
    # add default values
    cfg = om.merge(cfg, om.create({
        "attn_pdrop": 0.0,
        "resid_pdrop": 0.0,
        "emb_pdrop": 0.0,
        "attn_impl": "flash",
        "rotary_emb_base": 10000
    }))


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
    tolerance = 1e-3
    diff = abs(s1 - s2)
    try:
        assert diff < tolerance
    except:
        print(f"[{desc}] failed! sums differ by {diff:.6f}")
        print(f"HF sum: {s1:.6f}")
        print(f"Composer sum: {s2:.6f}")
        return 
    print(f"[{desc}] passed! sums are within tolerance ({diff:.6f})")


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
    hf_loss = hf_model(input_ids, labels=input_ids).loss

    cfg = construct_example_cfg(model_size)
    composer_model = ComposerMosaicPythia(cfg)
    # rotary_emb.inv_freq can be missing
    composer_model.load_state_dict(torch.load(composer_pythia_path), strict=False)

    input_ids = input_ids.cuda()
    composer_model.half().cuda()
    hf_model.half().cuda()

    # Add model dtype and device info for debugging
    print(f"\nModel Information:")
    print(f"HF Model dtype: {next(hf_model.parameters()).dtype}")
    print(f"Composer Model dtype: {next(composer_model.parameters()).dtype}")
    print(f"Input device: {input_ids.device}\n")

    logits1 = hf_model(input_ids, labels=input_ids).logits.mean()
    logits2 = composer_model({"input_ids": input_ids})["logits"].mean()

    test_two_matrix(logits1, logits2, "HF vs. Composer")

    # Add this after the existing tests:
    print("\nGenerating sample outputs:")
    
    # Set up generation parameters
    max_new_tokens = 50
    temperature = 0.7
    
    # Generate from both models for comparison
    input_prompt = "The future of artificial intelligence will"
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt").cuda()
    
    # Generate with Composer model using HF generation utilities
    with torch.no_grad():
        outputs = composer_model({"input_ids": input_ids})
        
        # Create generation config
        gen_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # Manual generation loop
        current_ids = input_ids
        for _ in range(max_new_tokens):
            outputs = composer_model({"input_ids": current_ids})
            next_token_logits = outputs["logits"][:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Stop if we hit the EOS token
            if next_token[0][0] == tokenizer.eos_token_id:
                break
    
    print("\nInput prompt:", input_prompt)
    print("\nComposer model output:")
    print(tokenizer.decode(current_ids[0], skip_special_tokens=True))

