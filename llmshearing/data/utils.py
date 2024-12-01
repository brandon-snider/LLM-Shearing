import os
import boto3
from smart_open import open
from datasets import load_dataset

session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)
s3 = session.client("s3")


def download_contents(files):
    for file in files:
        s3_url = f"s3://softwareheritage/content/{file['blob_id']}"
        with open(
            s3_url, "rb", compression=".gz", transport_params={"client": s3}
        ) as fin:
            file["content"] = fin.read().decode(file["src_encoding"])

    return {"files": files}


if __name__ == "__main__":
    ds = load_dataset(
        "bigcode/the-stack-v2-train-smol-ids", split="train", streaming=True
    )
    ds = ds.map(lambda row: download_contents(row["files"]))
    ds_iter = iter(ds)
    for row in ds_iter:
        for file in row["files"]:
            print(file["content"])
        break
