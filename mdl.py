import os
import modal

cuda_version = "11.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.9")
    .apt_install("git")
    .pip_install("torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2")
    .pip_install("packaging", "zstandard", "ninja", "wheel", "setuptools")
    .pip_install("composer==0.16.3", "llm-foundry==0.3.0", "fire")
    .apt_install("clang")
    .pip_install("flash-attn==1.0.3.post0")
    .pip_install("transformers==4.33.0")
)

app = modal.App("pruning")

volume = modal.Volume.from_name("pruning-vol", create_if_missing=True)


@app.function(image=image, volumes={"/pruning-vol": volume})
def f():
    print(os.listdir("/pruning-vol"))


@app.local_entrypoint()
def main():
    # with volume.batch_upload() as upload:
    #     upload.put_directory("models/pythia-14m-composer", "models/pythia-14m-composer")
    f.remote()
