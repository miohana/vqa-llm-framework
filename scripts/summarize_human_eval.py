import json
import pathlib
import argparse
import pandas as pd


def summarize_human_eval(frame: pd.DataFrame) -> dict:
    summary = {
        "faithfulness": frame["faithfulness"].mean(),
        "relevancy": frame["relevancy"].mean()
    }
    normalized_frame = (frame-1.0)/(5.0-1.0)
    return {
        **summary,
        "normalized_faithfulness": normalized_frame["faithfulness"].mean(),
        "normalized_relevancy": normalized_frame["relevancy"].mean()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize Human Evaluation"
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path, required=True, help="Human Eval Folder"
    )
    args = parser.parse_args()
    data_path: pathlib.Path = args.data

    assert data_path.is_dir(), "--data should be a valid directory"

    summary = {}
    for path in data_path.glob("*.csv"):
        frame = pd.read_csv(path)
        results = summarize_human_eval(frame)
        summary[path.stem] = results

    with data_path.joinpath("summary.json").open("w+") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
