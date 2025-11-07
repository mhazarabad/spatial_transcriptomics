import json
from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/visium_hne_subset.h5ad")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_spatial_data(
    filepath: Path
):
    """
    params:
        filepath: Path to the Visium H&E spatial transcriptomics dataset
    returns:
        expression: DataFrame of expression values
        coords: DataFrame of spatial coordinates
        metadata: DataFrame of metadata
    """
    adata = ad.read_h5ad(filepath)

    if "spatial" not in adata.obsm:
        raise ValueError("AnnData object is missing")

    expression = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        index=adata.obs_names,
        columns=adata.var_names,
    )
    coords = pd.DataFrame(adata.obsm["spatial"], columns=["x", "y"], index=adata.obs_names)
    metadata = adata.obs.copy()

    return expression, coords, metadata


def quality_control(
    expression: pd.DataFrame,
    coords: pd.DataFrame,
    metadata: pd.DataFrame,
    min_genes: int = 2000,
    min_counts: int = 10000,
    max_pct_mt: float = 30.0,
):
    """
    params:
        expression: DataFrame of expression values
        coords: DataFrame of spatial coordinates
        metadata: DataFrame of metadata
        min_genes: Minimum number of genes detected per spot
        min_counts: Minimum total counts per spot
        max_pct_mt: Maximum percentage of mitochondrial reads per spot
    returns:
        expression: DataFrame of expression values
        coords: DataFrame of spatial coordinates
        metadata: DataFrame of metadata
    """
    if not {"n_genes_by_counts", "total_counts", "pct_counts_mt", "in_tissue"}.issubset(metadata.columns):
        raise ValueError("Required QC columns are missing from the metadata.")

    keep_spots = (
        (metadata["in_tissue"] == 1)
        & (metadata["n_genes_by_counts"] >= min_genes)
        & (metadata["total_counts"] >= min_counts)
        & (metadata["pct_counts_mt"] <= max_pct_mt)
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # histograms of the QC metrics
    axes[0].hist(metadata["n_genes_by_counts"], bins=40, edgecolor="black")
    axes[0].axvline(min_genes, color="red", linestyle="--", label=f"≥ {min_genes}")
    axes[0].set_title("Genes detected per spot")
    axes[0].set_xlabel("n_genes_by_counts")
    axes[0].legend()
    # histogram of the total counts per spot
    axes[1].hist(metadata["total_counts"], bins=40, edgecolor="black")
    axes[1].axvline(min_counts, color="red", linestyle="--", label=f"≥ {min_counts}")
    axes[1].set_title("Total counts per spot")
    axes[1].set_xlabel("total_counts")
    axes[1].legend()
    # histogram of the mitochondrial percentage per spot
    axes[2].hist(metadata["pct_counts_mt"], bins=40, edgecolor="black")
    axes[2].axvline(max_pct_mt, color="red", linestyle="--", label=f"≤ {max_pct_mt}")
    axes[2].set_title("Mitochondrial percentage")
    axes[2].set_xlabel("pct_counts_mt")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "qc_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    return (
        expression.loc[keep_spots],
        coords.loc[keep_spots],
        metadata.loc[keep_spots],
    )


def normalize_expression(
    expression: pd.DataFrame,
):
    """
    params:
        expression: DataFrame of expression values
    returns:
        expression_norm: DataFrame of normalized expression values
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(expression.values)
    expression_norm = pd.DataFrame(scaled, index=expression.index, columns=expression.columns)
    return expression_norm

def select_variable_genes(
    expression: pd.DataFrame,
    n_top_genes: int = 50,
):
    """Select highly variable genes.
    params:
        expression: DataFrame of expression values
        n_top_genes: Number of top variable genes to select
    returns:
        expression_var: DataFrame of variable genes
    """
    gene_vars = expression.var(axis=0)
    top_genes = gene_vars.nlargest(n_top_genes).index.tolist()
    
    return expression[top_genes]

def spatial_clustering(
    expression: pd.DataFrame,
    coords: pd.DataFrame,
    n_clusters: int = 4,
):
    """Perform spatial clustering.
    params:
        expression: DataFrame of expression values
        coords: DataFrame of spatial coordinates
        n_clusters: Number of clusters to create
    returns:
        coords_clustered: DataFrame of spatial coordinates with cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=25)
    clusters = kmeans.fit_predict(expression)    
    coords_clustered = coords.copy()
    coords_clustered['cluster'] = clusters
    return coords_clustered

def find_marker_genes(
    expression: pd.DataFrame,
    clusters: np.ndarray,
    top_n: int = 10,
):
    """Identify marker genes for each cluster.
    params:
        expression: DataFrame of expression values
        clusters: Array of cluster labels
        top_n: Number of top markers to select per cluster
    returns:
        markers: Dictionary of marker genes for each cluster
    """
    markers = {}
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        pvalues = np.zeros(len(expression.columns))
        fold_changes = np.zeros(len(expression.columns))
        
        for gene in expression.columns:
            in_cluster = expression.loc[cluster_mask, gene]
            out_cluster = expression.loc[~cluster_mask, gene]
            
            if len(in_cluster) > 1 and len(out_cluster) > 1:
                stat, pval = stats.ranksums(in_cluster, out_cluster)
                pvalues.append(pval)
                mean_in = in_cluster.mean()
                mean_out = out_cluster.mean()
                fc = mean_in - mean_out if mean_out > 0 else 0
                fold_changes.append(fc)
            else:
                pvalues.append(1.0)
                fold_changes.append(0.0)
        
        results_df = pd.DataFrame({
            'gene': expression.columns,
            'pvalue': pvalues,
            'log_fold_change': fold_changes
        })
        
        # FDR correction (Benjamini-Hochberg)
        results_df = results_df.sort_values('pvalue')
        results_df['fdr'] = results_df['pvalue'] * len(results_df) / (np.arange(len(results_df)) + 1)
        results_df['fdr'] = results_df['fdr'].clip(upper=1.0)
        
        top_markers = results_df.nsmallest(top_n, 'pvalue')
        markers[f'Cluster_{cluster_id}'] = top_markers
    
    return markers

def visualize_results(
    coords_clustered: pd.DataFrame,
    expression: pd.DataFrame,
    markers: dict[str, pd.DataFrame],
):
    """
    params:
        coords_clustered: DataFrame of spatial coordinates with cluster labels
        expression: DataFrame of expression values
        markers: Dictionary of marker genes for each cluster
    """    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        coords_clustered['x'], 
        coords_clustered['y'], 
        c=coords_clustered['cluster'], 
        cmap='tab10', 
        s=100, 
        edgecolors='black', 
        linewidths=0.5, 
        alpha=0.8,
    )
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Spatial Distribution of Clusters')
    plt.grid(alpha=0.3)
    plt.savefig(RESULTS_DIR / 'spatial_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    top_genes = []
    for cluster_markers in markers.values():
        top_genes.extend(cluster_markers['gene'].head(5).tolist())
    top_genes = list(set(top_genes))[:20]
    
    if top_genes:
        marker_expression = expression[top_genes].T
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            marker_expression, 
            cmap='viridis', 
            cbar_kws={'label': 'Expression'}, 
            yticklabels=True, 
            xticklabels=False,
        )
        plt.xlabel('Spots')
        plt.ylabel('Marker Genes')
        plt.title('Expression Pattern of Top Marker Genes')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'marker_genes_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    all_markers = pd.concat(markers, names=['cluster', 'rank']).reset_index(level=0)
    all_markers.to_csv(RESULTS_DIR / 'cluster_markers.csv', index=False)

def evaluate_vs_reference(
    metadata: pd.DataFrame,
    clusters: np.ndarray,
):
    """
    params:
        metadata: DataFrame of metadata
        clusters: Array of cluster labels
    returns:
        metrics: Dictionary of metrics
    """
    metrics = {}
    if "cluster" in metadata.columns:
        reference_labels = metadata["cluster"].astype(str).values
        ari = adjusted_rand_score(reference_labels, clusters)
        contingency = pd.crosstab(reference_labels, clusters)
        contingency.to_csv(RESULTS_DIR / "cluster_contingency.csv")
        metrics["adjusted_rand_index"] = float(ari)
    return metrics

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed Visium dataset not found at {DATA_PATH}. "
        "Run `generate_test_data.py` first to download the subset."
    )
expression, coords, metadata = load_spatial_data(DATA_PATH)
expression_qc, coords_qc, metadata_qc = quality_control(expression, coords, metadata)

expression_norm = normalize_expression(expression_qc)
expression_var = select_variable_genes(expression_norm, n_top_genes=40)

coords_clustered = spatial_clustering(expression_var, coords_qc, n_clusters=6)
markers = find_marker_genes(expression_qc.loc[coords_clustered.index], coords_clustered['cluster'].values, top_n=15)
visualize_results(coords_clustered, expression_norm.loc[coords_clustered.index], markers)
metrics = evaluate_vs_reference(metadata_qc.loc[coords_clustered.index], coords_clustered["cluster"].values)
summary = {
    "input_file": str(DATA_PATH),
    "spots_qc": int(expression_qc.shape[0]),
    "genes_qc": int(expression_qc.shape[1]),
    "clusters": coords_clustered["cluster"].nunique(),
    "cluster_distribution": coords_clustered["cluster"].value_counts().to_dict(),
    **metrics,
}
(RESULTS_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))



