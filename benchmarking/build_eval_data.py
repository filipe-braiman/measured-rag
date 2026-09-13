"""Build the QASPER-based evaluation dataset and download its papers."""

import argparse
import json
import time

import requests
from datasets import load_dataset

from benchmarking.evaluation_config import load_evaluation_config

#Dataset Config
SEED = 1001 #previously tried 1001 7331, 401
SAMPLE_SIZE = 28 #previously tried 10, 15
MIN_QAS = 3 #minimum number of QAs per paper to be eligible for the evaluation subset (default: 3)

QASPER_REVISION = "d57bf25d29d3f089384b087ab24fb9bd077a9e09"
QASPER_VALIDATION_URL = (
    "https://huggingface.co/datasets/allenai/qasper/"
    f"resolve/{QASPER_REVISION}/qasper/validation/0000.parquet"
)
QASPER_TEST_URL = (
    "https://huggingface.co/datasets/allenai/qasper/"
    f"resolve/{QASPER_REVISION}/qasper/test/0000.parquet"
)


def build_eval_data():
    profile = load_evaluation_config()
    output_dir = profile.pdf_path
    qa_output_path = profile.qa_path
    output_dir.mkdir(parents=True, exist_ok=True)
    qa_output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "parquet",
        data_files={"validation": QASPER_VALIDATION_URL},
        split="validation",
    )
    eligible = dataset.filter(
        lambda row: (
            len(row["qas"]["question"]) >= MIN_QAS
            and len(row["full_text"]["section_name"]) > 0
        )
    )
    print(f"Eligible papers: {len(eligible)}")

    sample = eligible.shuffle(seed=SEED).select(range(SAMPLE_SIZE))
    qa_records = [{"id": row["id"], "qas": row["qas"]} for row in sample]
    with qa_output_path.open("w", encoding="utf-8") as file:
        json.dump(qa_records, file, ensure_ascii=False, indent=2)
    print(f"Saved QAs for {len(qa_records)} papers: {profile.qa_data_path}")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Qasper-RAG-Evaluation/1.0 "
                "(personal research evaluation)"
            )
        }
    )
    for row in sample:
        paper_id = row["id"].removeprefix("arXiv:")
        output_path = output_dir / f"{paper_id}.pdf"
        if output_path.exists():
            print(f"Already downloaded: {paper_id}")
            continue
        response = session.get(
            f"https://arxiv.org/pdf/{paper_id}.pdf",
            timeout=90,
        )
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"Downloaded content for {paper_id} is not a PDF.")
        output_path.write_bytes(response.content)
        print(f"Downloaded: {paper_id}")
        time.sleep(3)


def build_parser():
    return argparse.ArgumentParser(
        description="Build the fixed QASPER evaluation subset."
    )


def main(argv=None):
    build_parser().parse_args(argv)
    build_eval_data()


if __name__ == "__main__":
    main()
