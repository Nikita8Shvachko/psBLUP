import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from improved_rrblup import improved_rrblup
from psBLUP import psblup


def load_chickpea_data(
    data_path: str = "./nuts_data",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load chickpea phenotype and genotype data"""

    print("Loading chickpea data...")

    # Load phenotype data
    pheno_file = os.path.join(data_path, "pheno_2016_VIRVFVIR_421_408_synchro.csv")
    pheno_data = pd.read_csv(pheno_file)

    # Load genotype data
    geno_file = os.path.join(
        data_path, "total_df_for_aio_chickpea_28042016_synchro.csv"
    )
    geno_data = pd.read_csv(geno_file, index_col=0)

    print("Raw data loaded:")
    print(f"Phenotype data dimensions: {pheno_data.shape}")
    print(f"Genotype data dimensions: {geno_data.shape}")

    return pheno_data, geno_data


def prepare_chickpea_data(
    pheno_data: pd.DataFrame, geno_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Prepare and match chickpea phenotype and genotype data"""

    print("\n=============== Data Preparation ===============")

    # Extract TSW trait data
    tsw_data = pheno_data[["SNP ID", "TSW"]].copy()
    tsw_data.columns = ["sample_id", "TSW"]

    # Remove rows with missing TSW values
    tsw_data = tsw_data.dropna(subset=["TSW"])

    # Set sample IDs as index for TSW data
    tsw_data = tsw_data.set_index("sample_id")

    # Find common samples
    common_samples = list(set(tsw_data.index) & set(geno_data.index))

    print("=============== Data Matching ===============")
    print(f"Samples in phenotype data: {len(tsw_data)}")
    print(f"Samples in genotype data: {len(geno_data)}")
    print(f"Common samples: {len(common_samples)}")

    if len(common_samples) == 0:
        raise ValueError("No common samples found between phenotype and genotype data!")

    # Filter both datasets to common samples
    snp_data_full = geno_data.loc[common_samples]
    met_data = tsw_data.loc[common_samples]
    N = len(common_samples)

    print("Full data dimensions:")
    print(f"SNP data: {snp_data_full.shape}")
    print(f"TSW data: {met_data.shape}")

    return met_data, snp_data_full, N


def advanced_quality_control(
    snp_data: pd.DataFrame, met_data: pd.DataFrame
) -> pd.DataFrame:
    """Perform advanced quality control and SNP selection"""

    print("\n=============== Advanced SNP Filtering ===============")

    snp_data_qc = snp_data.copy()

    # Remove SNPs with no variation
    no_variation = []
    for col in snp_data_qc.columns:
        unique_vals = snp_data_qc[col].dropna().unique()
        if len(unique_vals) <= 1:
            no_variation.append(col)

    if len(no_variation) > 0:
        print(f"Removing {len(no_variation)} SNPs with no variation")
        snp_data_qc = snp_data_qc.drop(columns=no_variation)

    # Remove SNPs with too many missing values (>10%)
    missing_threshold = 0.1
    missing_prop = snp_data_qc.isnull().sum() / len(snp_data_qc)
    high_missing = missing_prop > missing_threshold

    if high_missing.sum() > 0:
        print(
            f"Removing {high_missing.sum()} SNPs with >{missing_threshold*100}% missing values"
        )
        snp_data_qc = snp_data_qc.loc[:, ~high_missing]

    print(f"After quality control: {snp_data_qc.shape[1]} SNPs remaining")

    return snp_data_qc


def select_informative_snps(
    snp_data: pd.DataFrame, tsw_values: pd.Series, max_snps: int = 50000
) -> pd.DataFrame:
    """Select most informative SNPs based on correlation with TSW"""

    print("\n=============== SNP Selection for Analysis ===============")

    # Calculate correlation with TSW trait
    correlations = {}
    for col in snp_data.columns:
        try:
            corr, _ = pearsonr(
                snp_data[col].dropna(), tsw_values[snp_data[col].dropna().index]
            )
            if not np.isnan(corr):
                correlations[col] = abs(corr)
        except (ValueError, IndexError):
            continue

    # Select top SNPs by absolute correlation
    if len(correlations) > max_snps:
        top_snps = sorted(
            correlations.keys(), key=lambda x: correlations[x], reverse=True
        )[:max_snps]
        snp_data_selected = snp_data[top_snps]
        print(f"Selected top {max_snps} most informative SNPs for analysis")
    else:
        snp_data_selected = snp_data[
            [col for col in snp_data.columns if col in correlations]
        ]
        print(f"Using all {snp_data_selected.shape[1]} remaining SNPs")

    print(f"Final SNP data dimensions: {snp_data_selected.shape}")
    print(f"SNP missing values: {snp_data_selected.isnull().sum().sum()}")

    return snp_data_selected


def create_sparse_proximity_matrix(
    snp_data: pd.DataFrame, correlation_threshold: float = 0.1
) -> np.ndarray:
    """Create sparse proximity matrix for computational efficiency"""

    print("\n=============== Creating Proximity Matrix ===============")

    # Calculate correlation matrix between SNPs
    cor_matrix = snp_data.corr().fillna(0).values

    # Apply threshold to create sparse proximity matrix
    proximity_matrix = cor_matrix**2
    proximity_matrix[np.abs(cor_matrix) < correlation_threshold] = 0

    # Add row sums as diagonal (Laplacian matrix requirement)
    row_sums = np.sum(proximity_matrix, axis=1)
    np.fill_diagonal(proximity_matrix, row_sums)

    sparsity = np.sum(proximity_matrix == 0) / proximity_matrix.size * 100
    print(f"Proximity matrix created: {proximity_matrix.shape}")
    print(f"Sparsity: {sparsity:.1f}% zeros")

    return proximity_matrix


def run_chickpea_analysis(
    met_data: pd.DataFrame, snp_data: pd.DataFrame, proximity_matrix: np.ndarray
) -> Tuple[Dict, Dict]:
    """Run combined psBLUP and improved rrBLUP analysis for TSW"""

    print(
        "\n=============== Running psBLUP and rrBLUP Analysis for TSW ==============="
    )

    n_runs = 10
    l2s = [
        1,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    ]  # Reduced range and fewer values
    percent_train_data = [0.4, 0.5, 0.6, 0.7]  # Fewer training proportions

    print("Analysis parameters:")
    print(f"  Runs: {n_runs}")
    print(f"  L2 values: {list(l2s)}")
    print(f"  Training proportions: {percent_train_data}")

    complete_samples = met_data["TSW"].count()
    print(
        f"\nAnalyzing TSW trait with {complete_samples} complete samples and {snp_data.shape[1]} SNPs"
    )

    # Variables to store results
    results = {}

    # Check TSW completeness
    if complete_samples < 30:
        raise ValueError(
            "Insufficient samples with TSW data for robust analysis (need at least 30)"
        )

    # Run improved rrBLUP first
    print("\n--- Running Improved rrBLUP ---")
    try:
        rrblup_result = improved_rrblup(
            Y=met_data,
            X=snp_data,
            log_transform_x=True,
            percent_train_data=percent_train_data,
            runs=n_runs,
            method="REML",
        )
        # print("Improved rrBLUP is not run")
        results["rrblup"] = rrblup_result["TSW"]
        rrblup_accuracy_matrix = rrblup_result["TSW"]["accuracy"]
        avg_rrblup = np.mean(rrblup_accuracy_matrix[:, 0])

        print(f"✓ Improved rrBLUP completed - Average accuracy: {avg_rrblup:.3f}")
        results["rrblup_success"] = True
        results["avg_rrblup"] = avg_rrblup

    except Exception as e:
        print(f"✗ Improved rrBLUP Analysis failed: {e}")
        results["rrblup_success"] = False
        results["avg_rrblup"] = np.nan

    # Run psBLUP
    print("\n--- Running psBLUP ---")
    try:
        psblup_result = psblup(
            Y=met_data,
            X=snp_data,
            proximity_matrix=proximity_matrix,
            log_transform_x=True,
            percent_train_data=percent_train_data,
            l2s=l2s,
            runs=n_runs,
        )

        results["psblup"] = psblup_result["TSW"]
        accuracy_matrix = psblup_result["TSW"]["accuracy"]
        psblup_averages = np.mean(accuracy_matrix, axis=0)
        avg_best_psblup = np.max(psblup_averages)
        best_l2_idx = np.argmax(psblup_averages)
        best_l2 = l2s[best_l2_idx]

        print(
            f"✓ psBLUP completed - Best average accuracy: {avg_best_psblup:.3f} (l2={best_l2})"
        )
        results["psblup_success"] = True
        results["avg_best_psblup"] = avg_best_psblup
        results["best_l2"] = best_l2
        results["psblup_averages"] = psblup_averages
        results["l2s"] = l2s

    except Exception as e:
        print(f"✗ psBLUP Analysis failed: {e}")
        results["psblup_success"] = False
        results["avg_best_psblup"] = np.nan

    # Store analysis parameters
    analysis_params = {
        "n_runs": n_runs,
        "l2s": l2s,
        "percent_train_data": percent_train_data,
        "complete_samples": complete_samples,
        "n_snps": snp_data.shape[1],
    }

    return results, analysis_params


def summarize_chickpea_results(results: Dict, analysis_params: Dict) -> None:
    """Summarize and display analysis results"""

    print("\n====== CHICKPEA TSW ANALYSIS RESULTS ======")

    if results.get("psblup_success", False) or results.get("rrblup_success", False):
        if results.get("rrblup_success", False):
            print(f"Average improved rrBLUP accuracy:      {results['avg_rrblup']:.3f}")

        if results.get("psblup_success", False):
            print(f"Average Best psBLUP accuracy: {results['avg_best_psblup']:.3f}")

            if results.get("rrblup_success", False):
                improvement = results["avg_best_psblup"] - results["avg_rrblup"]
                print(f"Overall improvement:          {improvement:+.3f}")

                if improvement > 0:
                    print("✓ psBLUP shows improvement over improved rrBLUP")
                else:
                    print("✗ psBLUP does not improve over improved rrBLUP")

            # Show psBLUP performance by l2 value
            print("\npsBLUP Performance by l2 value:")
            for i, l2_val in enumerate(results["l2s"]):
                l2_avg = results["psblup_averages"][i]
                if results.get("rrblup_success", False):
                    improvement = l2_avg - results["avg_rrblup"]
                    print(
                        f"  l2 = {l2_val:3d}: {l2_avg:.4f} (improvement: {improvement:+.4f})"
                    )
                else:
                    print(f"  l2 = {l2_val:3d}: {l2_avg:.4f}")
    else:
        print("✗ Both analyses failed")

    print("\n=============== ANALYSIS SUMMARY ===============")
    print(f"Samples: {analysis_params['complete_samples']}")
    print(f"SNPs used: {analysis_params['n_snps']}")
    print(f"Analysis runs: {analysis_params['n_runs']}")
    print(f"Training proportions: {len(analysis_params['percent_train_data'])}")
    print(f"L2 penalty values tested: {len(analysis_params['l2s'])}")


def save_chickpea_results(results: Dict, analysis_params: Dict) -> None:
    """Save analysis results to files"""

    # Create output directory
    output_dir = "./output_chickpea_py"
    os.makedirs(output_dir, exist_ok=True)

    if results.get("psblup_success", False) or results.get("rrblup_success", False):

        # Save summary results
        summary_data = []

        if results.get("rrblup_success", False):
            summary_data.append(
                {
                    "Method": "Improved_rrBLUP",
                    "Average_Accuracy": results["avg_rrblup"],
                    "SNPs_Used": analysis_params["n_snps"],
                    "Analysis_Details": (
                        f"Runs:{analysis_params['n_runs']} "
                        f"Train_Props:{len(analysis_params['percent_train_data'])}"
                    ),
                }
            )

        if results.get("psblup_success", False):
            summary_data.append(
                {
                    "Method": "psBLUP_Best",
                    "Average_Accuracy": results["avg_best_psblup"],
                    "SNPs_Used": analysis_params["n_snps"],
                    "Analysis_Details": f"L2s:{len(analysis_params['l2s'])} Runs:{analysis_params['n_runs']}",
                }
            )

            if results.get("rrblup_success", False):
                improvement = results["avg_best_psblup"] - results["avg_rrblup"]
                summary_data.append(
                    {
                        "Method": "Improvement",
                        "Average_Accuracy": improvement,
                        "SNPs_Used": analysis_params["n_snps"],
                        "Analysis_Details": "psBLUP vs Improved_rrBLUP",
                    }
                )

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, "chickpea_tsw_python_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary results saved to '{summary_file}'")

        # Save detailed results if available
        if results.get("psblup_success", False):
            psblup_detailed = []
            accuracy_matrix = results["psblup"]["accuracy"]
            for i, train_prop in enumerate(analysis_params["percent_train_data"]):
                for j, l2_val in enumerate(analysis_params["l2s"]):
                    if i < accuracy_matrix.shape[0] and j < accuracy_matrix.shape[1]:
                        psblup_detailed.append(
                            {
                                "trait": "TSW",
                                "training_proportion": train_prop,
                                "l2_value": l2_val,
                                "accuracy": accuracy_matrix[i, j],
                            }
                        )

            psblup_df = pd.DataFrame(psblup_detailed)
            psblup_file = os.path.join(output_dir, "chickpea_tsw_psblup_detailed.csv")
            psblup_df.to_csv(psblup_file, index=False)
            print(f"Detailed psBLUP results saved to '{psblup_file}'")

        if results.get("rrblup_success", False):
            rrblup_detailed = []
            accuracy_matrix = results["rrblup"]["accuracy"]
            for i, train_prop in enumerate(analysis_params["percent_train_data"]):
                if i < accuracy_matrix.shape[0]:
                    rrblup_detailed.append(
                        {
                            "trait": "TSW",
                            "training_proportion": train_prop,
                            "accuracy": accuracy_matrix[i, 0],
                        }
                    )

            rrblup_df = pd.DataFrame(rrblup_detailed)
            rrblup_file = os.path.join(output_dir, "chickpea_tsw_rrblup_detailed.csv")
            rrblup_df.to_csv(rrblup_file, index=False)
            print(f"Detailed improved rrBLUP results saved to '{rrblup_file}'")

    else:
        print("No results to save - both analyses failed")


def main():
    """Main function to run the chickpea TSW analysis"""

    print("##### Optimized Chickpea TSW psBLUP Analysis - Python Version ####")
    print("Using improved rrBLUP with REML estimation for better R compatibility")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)

    try:
        # Load data
        pheno_data, geno_data = load_chickpea_data()

        # Prepare data
        met_data, snp_data_full, N = prepare_chickpea_data(pheno_data, geno_data)

        # Quality control
        snp_data_qc = advanced_quality_control(snp_data_full, met_data)

        # Select informative SNPs
        snp_data_selected = select_informative_snps(snp_data_qc, met_data["TSW"])

        # Create proximity matrix
        proximity_matrix = create_sparse_proximity_matrix(snp_data_selected)

        # Run analysis
        results, analysis_params = run_chickpea_analysis(
            met_data, snp_data_selected, proximity_matrix
        )

        # Summarize results
        summarize_chickpea_results(results, analysis_params)

        # Save results
        save_chickpea_results(results, analysis_params)

        print("\n✅ Optimized Chickpea TSW Analysis Complete (Python Version)")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
