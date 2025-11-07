#!/usr/bin/env python3
"""Download and preprocess a real Visium spatial transcriptomics dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import anndata as ad
import numpy as np
import pandas as pd
import requests
from scipy import sparse

FIGSHARE_URL = "https://ndownloader.figshare.com/files/26098397"
RAW_FILENAME = "visium_hne_adata_full.h5ad"
PROCESSED_FILENAME = "visium_hne_subset.h5ad"
SUMMARY_FILENAME = "visium_dataset_summary.json"
COORDS_FILENAME = "visium_spatial_coordinates.csv"

N_TOP_GENES = 3000
MAX_SPOTS = 2000
RANDOM_SEED = 42


def download_figshare(url: str, destination: Path) -> None:
    """Stream a file from Figshare to ``destination``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 14):
                if chunk:
                    fh.write(chunk)


def _sparse_variance(matrix: sparse.spmatrix) -> np.ndarray:
    """Compute column-wise variance for a sparse matrix."""
    matrix = matrix.tocsr()
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    sq_mean = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    return sq_mean - mean**2


def _dense_variance(matrix: np.ndarray) -> np.ndarray:
    """Column-wise variance for dense arrays."""
    return matrix.var(axis=0)


def _compute_top_gene_indices(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        variances = _sparse_variance(matrix)
    else:
        variances = _dense_variance(matrix)

    n_genes = min(N_TOP_GENES, variances.shape[0])
    top_idx = np.argpartition(variances, -n_genes)[-n_genes:]
    top_idx.sort()
    return top_idx


def subset_ann_data(adata: ad.AnnData) -> Tuple[ad.AnnData, int, int]:
    """Subset the AnnData object to the most informative spots and genes."""
    adata = adata.copy()
    adata.var_names_make_unique()

    top_gene_idx = _compute_top_gene_indices(adata.X)
    adata = adata[:, top_gene_idx].copy()

    original_spots = adata.n_obs
    if adata.n_obs > MAX_SPOTS:
        rng = np.random.default_rng(RANDOM_SEED)
        keep_idx = np.sort(rng.choice(adata.n_obs, size=MAX_SPOTS, replace=False))
        adata = adata[keep_idx, :].copy()

    if sparse.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    return adata, original_spots, top_gene_idx.size


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_path = data_dir / RAW_FILENAME
    processed_path = data_dir / PROCESSED_FILENAME

    if not raw_path.exists():
        print(f"Downloading Visium H&E dataset to {raw_path} ...")
        download_figshare(FIGSHARE_URL, raw_path)
        print("Download complete.")
    else:
        print(f"Found existing file at {raw_path}, skipping download.")

    print("Loading raw AnnData file ...")
    adata = ad.read_h5ad(raw_path)

    if "spatial" not in adata.obsm:
        raise ValueError("Expected spatial coordinates in `adata.obsm['spatial']` but none were found.")

    print("Subsetting genes and spots for a lightweight benchmark ...")
    adata_subset, raw_spots, kept_genes = subset_ann_data(adata)

    adata_subset.write_h5ad(processed_path, compression="gzip")
    print(f"Processed dataset saved to {processed_path} ({adata_subset.n_obs} spots x {adata_subset.n_vars} genes).")

    coords_df = pd.DataFrame(
        adata_subset.obsm["spatial"],
        columns=["x", "y"],
        index=adata_subset.obs_names,
    )
    coords_df.to_csv(data_dir / COORDS_FILENAME, index_label="spot_id")

    summary = [
        {
            "source_url": FIGSHARE_URL,
            "raw_file": RAW_FILENAME,
            "processed_file": PROCESSED_FILENAME,
            "spots_raw": int(raw_spots),
            "spots_processed": int(adata_subset.n_obs),
            "genes_processed": int(adata_subset.n_vars),
        }
    ]
    (data_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary metadata to {SUMMARY_FILENAME} and spatial coordinates to {COORDS_FILENAME}.")


if __name__ == "__main__":
    main()

