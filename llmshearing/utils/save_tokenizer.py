import sys
from transformers import AutoTokenizer

def download_tokenizer(model_name, save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    model_name = sys.argv[1]
    save_path = sys.argv[2]
    download_tokenizer(model_name, save_path)