import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def parse(path):
    with open(path, "r") as g:
        for line in g:
            yield json.loads(line)


def get_df(path):
    return pd.DataFrame(list(parse(path)))


def filter_users_by_interactions(matrix, user_indices, min_items=2):
    user_interactions = matrix[user_indices]
    mask = np.array(user_interactions.getnnz(axis=1) >= min_items)
    return user_indices[mask]


def main(
    input_path,
    output_dir=None,
    dataset=None,
    sample_size=100000,
    test_size=0.2,
    min_ratings=2,
    quantile=0.95,
):
    input_path = Path(input_path)
    dataset = dataset or input_path.stem
    output_dir = Path(output_dir or dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {input_path}...")
    books = get_df(input_path)
    print(f"Total records loaded: {len(books)}")

    user_counts = books["reviewerID"].value_counts()
    item_counts = books["asin"].value_counts()

    top_users = user_counts[user_counts >= user_counts.quantile(quantile)].index
    top_items = item_counts[item_counts >= item_counts.quantile(quantile)].index

    print(f"Top {100 * (1 - quantile):.1f}% users: {len(top_users)}")
    print(f"Top {100 * (1 - quantile):.1f}% items: {len(top_items)}")

    filtered = books[
        books["reviewerID"].isin(top_users) & books["asin"].isin(top_items)
    ]
    print(f"Filtered records: {len(filtered)}")

    df = filtered.sample(n=min(sample_size, len(filtered)), random_state=42).copy()
    print(f"Sampled records: {len(df)}")

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df["user"] = user_enc.fit_transform(df["reviewerID"])
    df["item"] = item_enc.fit_transform(df["asin"])

    pickle.dump(
        list(user_enc.classes_), open(output_dir / f"{dataset}_user_enc_list.pkl", "wb")
    )
    pickle.dump(
        list(item_enc.classes_), open(output_dir / f"{dataset}_item_enc_list.pkl", "wb")
    )

    sparse_matrix = coo_matrix((df["overall"], (df["user"], df["item"]))).tocsr()
    sparse_matrix.data[:] = 1.0

    print("Creating train/test split...")
    all_users = np.arange(sparse_matrix.shape[0])
    train_users, test_users = train_test_split(
        all_users, test_size=test_size, random_state=42
    )

    train_users_filtered = filter_users_by_interactions(
        sparse_matrix, train_users, min_items=min_ratings
    )
    test_users_filtered = filter_users_by_interactions(
        sparse_matrix, test_users, min_items=min_ratings
    )

    print(f"Train users after filtering: {len(train_users_filtered)}")
    print(f"Test users after filtering: {len(test_users_filtered)}")

    train_mask = np.zeros(sparse_matrix.shape[0], dtype=bool)
    test_mask = np.zeros(sparse_matrix.shape[0], dtype=bool)
    train_mask[train_users_filtered] = True
    test_mask[test_users_filtered] = True

    train_matrix = sparse_matrix[train_mask]
    test_matrix = sparse_matrix[test_mask]

    pickle.dump(train_matrix, open(output_dir / f"{dataset}_train.pkl", "wb"))
    pickle.dump(test_matrix, open(output_dir / f"{dataset}_valid.pkl", "wb"))

    print("\n✅ Processing complete.")
    print(f"Dataset: {dataset}")
    print(f"Train users: {train_matrix.shape[0]}, Test users: {test_matrix.shape[0]}")
    print(
        f"Train interactions: {train_matrix.nnz}, Test interactions: {test_matrix.nnz}"
    )
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and filter Amazon Books dataset."
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to input JSON dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output (defaults to dataset name)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Optional dataset name override"
    )
    parser.add_argument(
        "--sample_size", type=int, default=100000, help="Max number of samples to keep"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test set proportion"
    )
    parser.add_argument(
        "--min_ratings",
        type=int,
        default=2,
        help="Minimum number of interactions per user",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Quantile cutoff for selecting top users/items (default: 0.95)",
    )
    args = parser.parse_args()
    main(**vars(args))
