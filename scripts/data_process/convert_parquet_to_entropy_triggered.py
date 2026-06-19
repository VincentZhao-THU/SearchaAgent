import argparse
import os

import pandas as pd


def make_entropy_triggered_prefix(question: str) -> str:
    question = question.strip()
    if question and question[-1] != "?":
        question += "?"
    return (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "Use <answer> and </answer> only when you are ready to provide the final answer. "
        "Search is controlled externally, so do not output <search> tags in your normal response. "
        "If later you are explicitly asked to provide a search query, output only the raw query text and nothing else. "
        f"Question: {question}\n"
    )


def convert_split(input_path: str, output_path: str) -> None:
    df = pd.read_parquet(input_path)

    required_columns = {"question", "prompt", "reward_model"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {sorted(missing)}")

    df = df.copy()
    df["prompt"] = df["question"].apply(
        lambda question: [{"role": "user", "content": make_entropy_triggered_prefix(question)}]
    )
    df["reward_model"] = df["reward_model"].apply(
        lambda reward_model: {
            **reward_model,
            "style": "rule_entropy_triggered",
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    for split in ("train", "test"):
        convert_split(
            input_path=os.path.join(args.input_dir, f"{split}.parquet"),
            output_path=os.path.join(args.output_dir, f"{split}.parquet"),
        )
