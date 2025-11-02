import argparse
import csv
import os
import random
from typing import List, Dict, Optional


def downsample_with_pandas(
    input_path: str,
    output_path: str,
    sample_size: int,
    seed: int,
    stratify_col: Optional[str] = None,
) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required for non-streaming downsampling. Install pandas or use --stream."
        ) from exc

    df = pd.read_csv(input_path)
    if sample_size > len(df):
        sample_size = len(df)

    if stratify_col and stratify_col in df.columns:
        try:
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required for stratified sampling. Install scikit-learn or omit --stratify-col."
            ) from exc

        # Use train_test_split to obtain an exact stratified sample of size n
        test_size = sample_size / len(df)
        _, df_sample = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df[stratify_col],
        )
    else:
        df_sample = df.sample(n=sample_size, random_state=seed)

    # Preserve column order
    df_sample.to_csv(output_path, index=False)


def reservoir_downsample_stream(
    input_path: str, output_path: str, sample_size: int, seed: int
) -> None:
    random.seed(seed)

    with open(input_path, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.reader(f_in)
        header = next(reader)

        reservoir: List[List[str]] = []
        count = 0
        for row in reader:
            count += 1
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                # Replace element with decreasing probability
                j = random.randint(1, count)
                if j <= sample_size:
                    reservoir[j - 1] = row

    # If the file had fewer rows than requested, reservoir will just contain all rows
    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        writer.writerows(reservoir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample a large CSV to a fixed number of rows."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of rows to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--stratify-col",
        type=str,
        default=None,
        help="Optional column name to stratify on (pandas mode only)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help=(
            "Use streaming reservoir sampling (lower memory, ignores --stratify-col). "
            "Recommended for very large files."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input
    output_path = args.output
    sample_size = args.n
    seed = args.seed
    stratify_col = args.stratify_col

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if args.stream:
        if stratify_col:
            print(
                "[warning] --stratify-col is ignored in --stream mode; proceeding without stratification."
            )
        reservoir_downsample_stream(
            input_path=input_path,
            output_path=output_path,
            sample_size=sample_size,
            seed=seed,
        )
    else:
        downsample_with_pandas(
            input_path=input_path,
            output_path=output_path,
            sample_size=sample_size,
            seed=seed,
            stratify_col=stratify_col,
        )

    print(
        f"Saved downsampled dataset with up to {sample_size} rows to: {output_path}"
    )


if __name__ == "__main__":
    main()


