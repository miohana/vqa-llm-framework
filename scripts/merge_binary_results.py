import pathlib
import argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge Framework Assessments into a single CSV file."
    )
    parser.add_argument(
        "--csv",
        type=pathlib.Path, required=True, help="CSV Results"
    )
    parser.add_argument(
        "--json",
        type=pathlib.Path, required=True, help="JSON with Binary Results"
    )
    args = parser.parse_args()

    binary_results = pd.read_json(args.json)
    binary_results = binary_results[["id", "faithfulness", "relevancy"]]
    binary_results = binary_results.rename(
        columns={
            "faithfulness": "binary_faithfulness",
            "relevancy": "binary_relevancy"
        }
    )
    results = pd.read_csv(args.csv)
    merged_results = pd.merge(results, binary_results, on="id")

    output_path: pathlib.Path = args.csv.parent
    merged_results.to_csv(
        output_path.joinpath(f"{args.csv.stem}-merged.csv"),
        index=False
    )
