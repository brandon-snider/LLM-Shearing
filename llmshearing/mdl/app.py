# modal run llmshearing/mdl/app.py

import subprocess
import time
import modal

cuda_version = "11.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.9")
    .apt_install("git")
    .pip_install("packaging", "zstandard", "ninja", "wheel", "setuptools")
    .run_commands(
        "pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118"
    )
    .apt_install("clang")
    .pip_install("flash-attn==1.0.3.post0")
    .apt_install("openssh-server")
    .run_commands(
        "cd root && git clone https://github.com/brandon-snider/LLM-Shearing.git && cd LLM-Shearing && pip install -r requirement.txt && pip install -e ."
    )
    # .pip_install("composer==0.16.3", "llm-foundry==0.3.0", "fire")
    .run_commands("mkdir /run/sshd")
    .copy_local_file("~/.ssh/modal.pub", "/root/.ssh/authorized_keys")
    .run_commands("pip install --upgrade datasets")
    .pip_install("transformers==4.40.1")
)

app = modal.App("pruning")

volume = modal.Volume.from_name("pruning-vol", create_if_missing=True)
# modal volume put pruning-vol data data
# mdoal volume rm -r pruning-vol data


timeout = 60 * 60 * 24  # 1 day


@app.function(
    image=image,
    timeout=timeout,
    volumes={"/root/pruning-vol": volume},
    # cpu=2,
    # memory=512,
)
def ssh_server():
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    with modal.forward(port=22, unencrypted=True) as tunnel:
        hostname, port = tunnel.tcp_socket
        connection_cmd = f"ssh -p {port} -i ~/.ssh/modal root@{hostname}"
        print(connection_cmd)
        time.sleep(timeout)  # keep alive until timeout or killed


# @app.function(image=image, volumes={"/pruning-vol": volume}, gpu="any")
# def f():
#     print(os.listdir("/pruning-vol"))


@app.local_entrypoint()
def main():
    ssh_server.remote()
