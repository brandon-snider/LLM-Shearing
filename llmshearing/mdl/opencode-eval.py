# Put a model on the volume: modal volume put pruning-vol out/pythia_160m/hf models/pythia_160m/hf
# Run the evaluation: modal run llmshearing/mdl/lm-eval.py

import modal
import subprocess
import os
import time

app = modal.App("opencode-eval")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    # .run_commands(f"huggingface-cli login --token {os.environ['HUGGINGFACE_TOKEN']}")
    .run_commands(
        "git clone --depth 1 https://github.com/brandon-snider/OpenCodeEval.git"
    )
    .run_commands("cd OpenCodeEval && pip install -r requirements-eval.txt")
    .run_commands("cd OpenCodeEval && pip install -r requirements.txt")
    .run_commands("pip install func_timeout")
    .apt_install("wget")
    .run_commands("cd OpenCodeEval/src/data && bash dataset.sh")
    .apt_install("openssh-server")
    .run_commands("mkdir /run/sshd")
    .copy_local_file("~/.ssh/modal.pub", "/root/.ssh/authorized_keys")
    .run_commands("mv OpenCodeEval root/")
)

volume = modal.Volume.from_name("pruning-vol", create_if_missing=True)

local_secret = modal.Secret.from_dict(
    {"HUGGINGFACE_TOKEN": os.environ["HUGGINGFACE_TOKEN"]}
)


@app.function(
    image=image,
    gpu="any",
    volumes={"/root/pruning-vol": volume},
    secrets=[local_secret],
    timeout=1800,  # 30 minutes
)
def evaluate():
    subprocess.run(
        [
            "python",
            "/OpenCodeEval/src/main.py",
            "--model_name",
            # "/pruning-vol/models/pythia_160m/hf",  # HF path or local path
            "EleutherAI/pythia-160m",  # HF path or local path
            # "--tokenizer_name",
            # "Qwen/Qwen2.5-Coder-0.5B",
            "--trust_remote_code",
            "--task",
            "HumanEval",
            "--prompt_type",
            "Completion",  # "Completion" or "Instruction"
            "--model_type",
            "Base",  # "Base" or "Chat"
            # "--time_out",
            # "3",
            # "--num_gpus",
            # "1",
            # "--num_workers",
            # "1",
            "--save_path",
            "/pruning-vol/evals/pythia-160m",
            # "--batch_size",
            # "164",
            # "--num_samples",
            # "1",
            # "max_tokens",
            # "2048",
            # "--temperature",
            # "0.0",
            # "--prompt_prefix",
            # "",
            # "--prompt_suffix",
            # "",
            # "--response_prefix",
            # "",
            # "--response_suffix",
            # "",
        ]
    )


timeout = 60 * 60 * 24  # 1 day


@app.function(
    image=image,
    timeout=timeout,
    volumes={"/root/pruning-vol": volume},
    secrets=[local_secret],
    # cpu=16,
    # memory=4096,
    gpu="any",
)
def ssh_server():
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    with modal.forward(port=22, unencrypted=True) as tunnel:
        hostname, port = tunnel.tcp_socket
        connection_cmd = f"ssh -p {port} -i ~/.ssh/modal root@{hostname}"
        print(connection_cmd)
        time.sleep(timeout)  # keep alive until timeout or killed


@app.local_entrypoint()
def main():
    # evaluate.remote()
    ssh_server.remote()


# accelerate launch /root/bigcode-evaluation-harness/main.py \
#     --model Qwen/Qwen2.5-Coder-0.5B \
#     --metric_output_path /root/pruning-vol/bigcode-evals/qwen-coder-0.5b/mbpp.json \
#     --precision bf16 \
#     --tasks mbpp \
#     --allow_code_execution \
#     --batch_size 1 \
#     --n_samples 1 \
#     --temperature 0.0 \
#     --max_length_generation 512 \
#     --limit 100

# accelerate launch /root/bigcode-evaluation-harness/main.py \
#     --model Qwen/Qwen2.5-Coder-0.5B \
#     --metric_output_path /root/pruning-vol/bigcode-evals/qwen-coder-0.5b/humaneval.json \
#     --precision bf16 \
#     --tasks humaneval \
#     --allow_code_execution \
#     --batch_size 1 \
#     --n_samples 1 \
#     --temperature 0.0 \
#     --max_length_generation 512 \
#     --do_sample false \
#     --limit 100
