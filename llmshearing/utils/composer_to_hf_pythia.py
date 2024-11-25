import sys
import torch
from omegaconf import OmegaConf as om
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import os

"""The file contains the util functions to convert the composer model to the huggingface model or vice versa."""

""" convert hf weights to composer weights and test the equivalence """


def get_key_map_from_hf_to_composer(num_layers):
    """get the keymap from hf to composer"""
    key_map = {}
    key_map.update(
        {
            "gpt_neox.embed_in.weight": "model.transformer.wte.weight",
            "gpt_neox.final_layer_norm.weight": "model.transformer.ln_f.weight",
            "gpt_neox.final_layer_norm.bias": "model.transformer.ln_f.bias",
            "embed_out.weight": "model.transformer.output.weight",
        }
    )
    for i in range(num_layers):
        key_map.update(
            {
                f"gpt_neox.layers.{i}.attention.query_key_value.weight": f"model.transformer.blocks.{i}.attn.query_key_value.weight",
                f"gpt_neox.layers.{i}.attention.query_key_value.bias": f"model.transformer.blocks.{i}.attn.query_key_value.bias",
                f"gpt_neox.layers.{i}.attention.dense.weight": f"model.transformer.blocks.{i}.attn.out_proj.weight",
                f"gpt_neox.layers.{i}.attention.dense.bias": f"model.transformer.blocks.{i}.attn.out_proj.bias",
                f"gpt_neox.layers.{i}.attention.rotary_emb.inv_freq": f"model.transformer.blocks.{i}.attn.rotary_emb.inv_freq",
                f"gpt_neox.layers.{i}.input_layernorm.weight": f"model.transformer.blocks.{i}.ln_1.weight",
                f"gpt_neox.layers.{i}.input_layernorm.bias": f"model.transformer.blocks.{i}.ln_1.bias",
                f"gpt_neox.layers.{i}.post_attention_layernorm.weight": f"model.transformer.blocks.{i}.ln_2.weight",
                f"gpt_neox.layers.{i}.post_attention_layernorm.bias": f"model.transformer.blocks.{i}.ln_2.bias",
                f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight": f"model.transformer.blocks.{i}.mlp.up_proj.weight",
                f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias": f"model.transformer.blocks.{i}.mlp.up_proj.bias",
                f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight": f"model.transformer.blocks.{i}.mlp.down_proj.weight",
                f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias": f"model.transformer.blocks.{i}.mlp.down_proj.bias",
            }
        )
    return key_map


def get_key_map_from_composer_to_hf(num_layers):
    """get kepmap from composer to hf"""
    return {
        value: key for key, value in get_key_map_from_hf_to_composer(num_layers).items()
    }


def get_layer_num_from_weights(weights):
    """get the layer num from weights name, works for both hf and composer weights"""
    max_layer_i = 0
    keyword = ["layers.", "blocks."]
    for key in weights:
        for key_word in keyword:
            if key_word in key:
                current_i = int(
                    key[key.index(key_word) + len(key_word) :].split(".")[0]
                )
                if current_i > max_layer_i:
                    max_layer_i = current_i
    return max_layer_i + 1


def save_hf_to_composer(hf_model_name_or_path, output_path):
    """Convert composer model to huggingface model"""
    model = AutoModelForCausalLM.from_pretrained(hf_model_name_or_path)
    hf_weights = model.state_dict()

    n_layers = get_layer_num_from_weights(hf_weights)
    key_map = get_key_map_from_hf_to_composer(n_layers)
    composer_state_dict = {}
    for key in hf_weights:
        if key in key_map:
            composer_state_dict[key_map[key]] = hf_weights[key]
        else:
            # rotary will be ignored
            print(f"key {key} not found in keymap")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(composer_state_dict, output_path)
    print(f"saved composer model to {output_path}")


def construct_hf_config(model_config: om = None):
    assert model_config is not None, "model config is None"
    model_class = model_config.pop("model_class")

    if model_class == "GPTNeoXForCausalLM":
        hf_model_name = "EleutherAI/pythia-14m"
        tokenizer_name = "EleutherAI/pythia-14m"
        config = AutoConfig.from_pretrained(hf_model_name)

    for key in model_config:
        setattr(config, key, model_config[key])

    return config, tokenizer_name


def save_composer_to_hf(composer_model_path, output_path=None, model_config: om = None):
    """convert composer ckpt's weights to huggingface"""

    weights = torch.load(composer_model_path)
    if "state" in weights:
        weights = weights["state"]["model"]
    num_layers = get_layer_num_from_weights(weights)
    keymap = get_key_map_from_composer_to_hf(num_layers)

    hf_weights = {keymap[key]: weights[key] for key in weights if "rotary" not in key}
    config, tokenizer_nanme = construct_hf_config(model_config)

    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(hf_weights, strict=False)
    model = model.half()
    model.save_pretrained(output_path, dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_nanme)
    tokenizer.save_pretrained(output_path)

    print(f"saved hf model to {output_path}")


if __name__ == "__main__":
    func = sys.argv[1]
    other_cli_args = sys.argv[2:]
    if func == "save_hf_to_composer":
        save_hf_to_composer(*other_cli_args)
    elif func == "save_composer_to_hf":
        composer_model_path, output_path, *other_args = sys.argv[2:]
        cli_cfg = om.from_cli(other_args)
        save_composer_to_hf(composer_model_path, output_path, cli_cfg)
    else:
        raise ValueError(f"func {func} not found")
