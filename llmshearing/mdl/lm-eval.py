# Put a model on the volume: modal volume put pruning-vol out/pythia_160m/hf models/pythia_160m/hf
# Run the evaluation: modal run llmshearing/mdl/lm-eval.py

import modal
import subprocess

app = modal.App("lm-eval")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        "git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness && cd lm-evaluation-harness && pip install -e ."
    )
)

volume = modal.Volume.from_name("pruning-vol", create_if_missing=True)


@app.function(image=image, gpu="A100", volumes={"/pruning-vol": volume})
def evaluate():
    subprocess.run(
        [
            "lm-eval",
            "--model",
            "hf",
            "--model_args",
            "pretrained=/pruning-vol/models/pythia_160m/hf",
            "--tasks",
            "lambada",
            "--output_path",
            "/pruning-vol/evals/pythia_160m/hf",
        ]
    )


@app.local_entrypoint()
def main():
    evaluate.remote()
