import os

import click
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm


@click.command()
@click.option("--assays-dir", type=click.Path(exists=True), required=True)
@click.option("--outputs-dir", type=click.Path(exists=True), required=True)
@click.option("--index-file", type=click.Path(exists=True), required=True)
@click.option("--split-all-scores", is_flag=True)
def main(assays_dir: str, outputs_dir: str, index_file: str, split_all_scores: bool) -> None:
    if split_all_scores:
        all_scores_file = os.path.join(outputs_dir, "all.csv")
        all_scores_df = pd.read_csv(all_scores_file)
        all_scores_df["DMS_id"] = all_scores_df["sequence_id"].apply(lambda x: x.split("+")[0])
        for assay_name in all_scores_df["DMS_id"].unique():
            assay_df = all_scores_df[all_scores_df.DMS_id == assay_name]
            assay_df.to_csv(os.path.join(outputs_dir, f"{assay_name}.csv"), index=False)

    index_df = pd.read_csv(index_file)
    spearmans = {}

    for DMS_id in tqdm(index_df["DMS_id"].unique(), desc="Scoring", ncols=80):
        output_file = os.path.join(outputs_dir, f"{DMS_id}.csv")
        if not os.path.exists(output_file):
            continue

        assay_file = os.path.join(assays_dir, f"{DMS_id}.csv")
        assay_df = pd.read_csv(assay_file)
        output_df = pd.read_csv(output_file)

        # Remove the DMS_id from the sequence id and keep only sequence index within the assay
        output_df["sequence_id"] = output_df["sequence_id"].apply(lambda x: x.split("+")[1]).astype(int)
        output_df = output_df.sort_values(by="sequence_id")

        assert len(assay_df) == len(output_df), "Assay and output files must have the same number of rows"

        y_true = assay_df["DMS_score"].values
        y_pred = output_df["log_likelihood"].values

        spearman_corr, _ = spearmanr(y_true, y_pred)
        spearmans[DMS_id] = spearman_corr

    index_df["spearman_corr"] = index_df["DMS_id"].apply(lambda x: spearmans.get(x, None))
    print(index_df[index_df["spearman_corr"].notna()][["DMS_id", "spearman_corr"]])

    spearman_df = index_df[index_df["spearman_corr"].notna()]
    aggregated_spearman = (
        spearman_df.groupby(["coarse_selection_type", "UniProt_ID"])["spearman_corr"]
        .mean()
        .groupby("coarse_selection_type")
        .mean()
        .mean()
        .item()
    )
    print(f"Aggregated Spearman: {aggregated_spearman}")


if __name__ == "__main__":
    main()
