# Put a model on the volume: modal volume put pruning-vol out/pythia_160m/hf models/pythia_160m/hf
# Run the evaluation: modal run llmshearing/mdl/lm-eval.py

import modal
import subprocess
import os
import time

app = modal.App("bigcode-eval")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        "git clone --depth 1 https://github.com/bigcode-project/bigcode-evaluation-harness.git && cd bigcode-evaluation-harness && pip install -e ."
    )
    .run_commands("mv bigcode-evaluation-harness root/")
    .run_commands(f"huggingface-cli login --token {os.environ['HUGGINGFACE_TOKEN']}")
    .apt_install("openssh-server")
    .run_commands("mkdir /run/sshd")
    .copy_local_file("~/.ssh/modal.pub", "/root/.ssh/authorized_keys")
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
            "accelerate",
            "launch",
            "/root/bigcode-evaluation-harness/main.py",
            "--model",
            # "/pruning-vol/models/pythia_160m/hf",  # HF path or local path
            "Qwen/Qwen2.5-Coder-0.5B",  # HF path or local path
            "--precision",
            "bf16",  # Default: fp32
            "--tasks",
            "mbpp",
            "--metric_output_path",
            # "/root/pruning-vol/evals/Qwen2.5-Coder-0.5B-Instruct/hf/humaneval.json",  # Directory should exist
            "--allow_code_execution",
            "--batch_size",
            "1",  # Default: 1 (should be as large as possible, but <= n_samples)
            "--n_samples",
            "1",  # Default: 10, Recommended: 200
            "--temperature",
            "0.1",  # Default: 0.2
            "--max_length_generation",
            "512",  # Default: 512
            "--limit",
            "100",  # Default: None
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
