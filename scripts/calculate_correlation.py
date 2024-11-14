import json
import pathlib
import argparse
import pandas as pd
from collections import defaultdict


def main(human_eval: pathlib.Path, results: pathlib.Path) -> dict:
    correlations = defaultdict(dict)
    for eval in human_eval.glob("*.csv"):
        pair_results_name = eval.stem + "-results-merged"
        pair_results = next(
            p for p in results.iterdir() if p.stem == pair_results_name
        )
        eval_df = pd.read_csv(eval).sort_values(by=["id"])
        results_df = pd.read_csv(pair_results).sort_values(by=["id"])
        for metric in ("faithfulness", "relevancy"):
            correlations[eval.stem][metric] = {
                f"{col}_{method}": eval_df[metric].corr(results_df[col], method=method)
                for col in results_df
                for method in ('pearson', 'spearman')
                if str(col) != "id"
            }

    return dict(correlations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate correlation between Human Evaluation and Framework Metrics"
    )
    parser.add_argument(
        "--human-eval",
        dest="human_eval", type=pathlib.Path, required=True, help="Folder with human evaluation CSV files"
    )
    parser.add_argument(
        "--results",
        type=pathlib.Path, required=True, help="Folder with framework assessment results"
    )
    args = parser.parse_args()

    correlations = main(args.human_eval, args.results)
    with args.human_eval.joinpath("correlations.json").open("w+") as fp:
        json.dump(correlations, fp, indent=2, ensure_ascii=False)
