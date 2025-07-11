import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from improved_rrblup import improved_rrblup

from psBLUP import psblup


def rrblup(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    log_transform_x: bool = False,
    percent_train_data: Optional[List[float]] = None,
    y_variables: Optional[List[str]] = None,
    x_variables: Optional[List[str]] = None,
    grouping_variables: Optional[List[str]] = None,
    runs: int = 5,
    alpha: float = 1.0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Ridge Regression BLUP (rrBLUP) implementation
    """

    # Argument validation
    if X is None:
        raise ValueError("No predictor variables have been provided")
    if Y is None:
        raise ValueError("No dependent variables have been provided")

    if not isinstance(log_transform_x, bool):
        raise ValueError("Acceptable values for 'log_transform_x': True or False")

    if y_variables is None:
        y_variables = list(Y.select_dtypes(include=[np.number]).columns)

    if x_variables is None:
        x_variables = list(X.select_dtypes(include=[np.number]).columns)

    if percent_train_data is None:
        percent_train_data = [1.0]
        runs = 1
        warnings.warn(
            "Since no training data will be used, only 1 run will be conducted."
        )

    objects_to_return = {}

    # Create copies to avoid modifying originals
    X_work = X.copy()
    Y_work = Y.copy()

    # Keep only relevant variables
    if grouping_variables:
        Y_work = Y_work[y_variables + grouping_variables]
    else:
        Y_work = Y_work[y_variables]
    X_work = X_work[x_variables]

    # Apply log transformation
    if log_transform_x:
        for var in y_variables:
            Y_work[var] = np.log(Y_work[var])

    # Handle grouping variables
    if grouping_variables:
        warnings.warn(
            "Grouping variables handling is simplified in this Python version"
        )
        Y_work = Y_work[y_variables]

    # Scale X variables
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_work), columns=X_work.columns, index=X_work.index
    )

    N = len(X_scaled)

    # Run rrBLUP for every response variable
    for y_var in y_variables:
        print(f"  Processing rrBLUP for variable: {y_var}")

        # Scale response
        y_scaler = StandardScaler()
        response = y_scaler.fit_transform(Y_work[[y_var]]).flatten()

        # Handle negative correlations (flip variables)
        predictors = X_scaled.copy()
        correlations = []
        for col in x_variables:
            corr, _ = pearsonr(response, predictors[col])
            correlations.append(corr)

        correlations = np.array(correlations)
        negative_correlations = np.where(correlations < 0)[0]
        neg_effects = [x_variables[i] for i in negative_correlations]

        for col in neg_effects:
            col_data = predictors[col]
            min_val, max_val = col_data.min(), col_data.max()
            predictors[col] = max_val + min_val - col_data

        # Create matrices for results
        accuracy_stats = np.full((len(percent_train_data), runs), np.nan)
        overall_accuracy = np.zeros((len(percent_train_data), 1))

        # Run accuracy evaluation
        j = 0
        while j < runs:
            accuracy_per_scenario = np.full((len(percent_train_data), 1), np.nan)

            for k, proportion in enumerate(percent_train_data):
                try:
                    # Select training and testing data
                    train_indices = np.random.choice(
                        N, size=int(np.round(proportion * N)), replace=False
                    )
                    test_indices = np.setdiff1d(np.arange(N), train_indices)

                    X_train = predictors.iloc[train_indices]
                    Y_train = response[train_indices]

                    if len(train_indices) == N:
                        X_test = X_train.copy()
                        Y_test = Y_train.copy()
                    else:
                        X_test = predictors.iloc[test_indices]
                        Y_test = response[test_indices]

                    # Fit Ridge regression
                    model = Ridge(alpha=alpha, fit_intercept=True)
                    model.fit(X_train[x_variables], Y_train)

                    # Get predictions
                    Y_pred = model.predict(X_test[x_variables])

                    # Calculate accuracy
                    correlation, _ = pearsonr(Y_test, Y_pred)
                    accuracy_per_scenario[k, 0] = np.round(correlation, 3)

                except Exception as e:
                    print(f"    Error in rrBLUP run {j+1}, proportion {k+1}: {e}")
                    break

                # Store accuracy for this run
                accuracy_stats[k, j] = accuracy_per_scenario[k, 0]

            j += 1
            overall_accuracy += np.nan_to_num(accuracy_per_scenario, nan=0)

        overall_accuracy = overall_accuracy / j

        objects_to_return[y_var] = {
            "accuracy": overall_accuracy,
            "stats": accuracy_stats,
        }

    return objects_to_return


def load_data(
    data_path: str = "psBLUP-main",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load metabolite, SNP, and SNP map data"""

    print("=============== Loading Data ===============")

    # Load data files
    met_data = pd.read_csv(os.path.join(data_path, "metabolites.csv"), index_col=0)
    snp_data = pd.read_csv(os.path.join(data_path, "SNPs.csv"), index_col=0)
    snp_map = pd.read_csv(os.path.join(data_path, "SNPmap.csv"), index_col=0)

    # Find common samples
    samples = list(set(met_data.index) & set(snp_data.index))
    snp_data = snp_data.loc[samples]
    met_data = met_data.loc[samples]
    N = len(samples)

    print(f"Common samples: {N}")
    print(f"SNP data dimensions: {snp_data.shape}")
    print(f"Metabolite data dimensions: {met_data.shape}")
    print(f"SNP missing values: {snp_data.isna().sum().sum()}")
    print(f"Metabolite missing values: {met_data.isna().sum().sum()}")

    return met_data, snp_data, snp_map


def quality_control(
    met_data: pd.DataFrame, snp_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Perform quality control on the data"""

    print("=============== Quality Control ===============")

    # Remove SNPs with no variation
    no_variation = []
    for col in snp_data.columns:
        unique_vals = snp_data[col].dropna().unique()
        if len(unique_vals) <= 1:
            no_variation.append(col)

    if len(no_variation) > 0:
        print(f"Removing {len(no_variation)} SNPs with no variation")
        snp_data = snp_data.drop(columns=no_variation)

    # Filter metabolites with sufficient sample sizes
    complete_per_trait = met_data.count()
    print("\nMetabolite completeness:")
    print(f"Min complete samples per trait: {complete_per_trait.min()}")
    print(f"Max complete samples per trait: {complete_per_trait.max()}")

    # Select traits with at least 40 complete samples
    min_samples = 40
    good_traits = complete_per_trait[complete_per_trait >= min_samples].index.tolist()
    print(f"Traits with >={min_samples} complete samples: {len(good_traits)}")

    if len(good_traits) == 0:
        raise ValueError("No traits have sufficient sample size for analysis")

    # Select subset for analysis
    selected_traits = good_traits[:5]
    met_data_subset = met_data[selected_traits]

    print("Selected traits for analysis:")
    for trait in selected_traits:
        complete_count = met_data_subset[trait].count()
        print(f"  {trait}: {complete_count} complete samples")

    return met_data_subset, snp_data, selected_traits


def create_proximity_matrix(
    snp_data: pd.DataFrame, snp_map: pd.DataFrame
) -> np.ndarray:
    """Create proximity matrix for psBLUP"""

    print("=============== Creating Proximity Matrix ===============")

    # Keep only SNPs that remain after quality filtering
    snp_map = snp_map.loc[snp_data.columns]

    # Order by chromosome and position
    snp_map = snp_map.sort_values(["chromosome", "position"])
    snp_data = snp_data[snp_map.index]
    n_chromosomes = snp_map["chromosome"].nunique()

    print(f"Number of chromosomes: {n_chromosomes}")

    # Create chromosome-specific correlation matrices
    matrices = []
    for chrom in sorted(snp_map["chromosome"].unique()):
        chrom_snps = snp_map[snp_map["chromosome"] == chrom].index
        chrom_data = snp_data[chrom_snps]
        positions = snp_map.loc[chrom_snps, "position"].values

        # Distance matrix
        dist_matrix = np.abs(positions[:, np.newaxis] - positions[np.newaxis, :])

        # Correlation matrix
        corr_matrix = chrom_data.corr().fillna(0).values

        # Proximity matrix: squared correlation within 10kb distance
        prox_matrix = (corr_matrix**2) * (dist_matrix < 10000)
        matrices.append(prox_matrix)

    # Create block diagonal matrix
    from scipy.linalg import block_diag

    proximity_matrix = block_diag(*matrices)

    print(f"Proximity matrix created: {proximity_matrix.shape}")

    return proximity_matrix


def run_analysis(
    met_data: pd.DataFrame,
    snp_data: pd.DataFrame,
    proximity_matrix: np.ndarray,
    selected_traits: List[str],
) -> Tuple[Dict, Dict]:
    """Run combined psBLUP and rrBLUP analysis"""

    print("=============== Running psBLUP and rrBLUP Analysis ===============")

    all_results = {}
    all_rrblup_results = {}

    # Analysis parameters - synchronized with R script
    n_runs = 20
    l2s = np.arange(10, 190, 10)  # seq(10, 180, by = 10) in R
    percent_train_data = np.arange(0.3, 0.85, 0.1)  # seq(0.3, 0.8, by = 0.1) in R

    # Analyze each trait separately
    for trait in selected_traits:
        print(f"\n{'='*50}")
        print(f"Analyzing trait: {trait}")
        print(f"{'='*50}")

        # Get complete samples for this trait
        trait_data = met_data[[trait]].dropna()
        complete_samples = trait_data.index

        trait_subset = trait_data
        snp_subset = snp_data.loc[complete_samples]

        n_complete = len(trait_subset)
        print(f"  Complete samples: {n_complete}")

        if n_complete >= 5:
            psblup_success = False
            rrblup_success = False
            avg_rrblup = np.nan
            avg_best_psblup = np.nan

            # Run improved rrBLUP first
            print("\n  --- Running improved rrBLUP ---")
            try:
                rrblup_result = improved_rrblup(
                    Y=trait_subset,
                    X=snp_subset,
                    log_transform_x=True,
                    percent_train_data=list(percent_train_data),
                    runs=n_runs,
                    method="REML",
                )

                all_rrblup_results[trait] = rrblup_result[trait]
                rrblup_success = True

                # Get rrBLUP results
                rrblup_accuracy_matrix = rrblup_result[trait]["accuracy"]
                avg_rrblup = np.mean(rrblup_accuracy_matrix[:, 0])
                print(
                    f"  ✓ Improved rrBLUP completed - Average accuracy: {avg_rrblup:.3f}"
                )

            except Exception as e:
                print(f"  ✗ Improved rrBLUP Analysis failed: {e}")

            # Run psBLUP
            print("\n  --- Running psBLUP ---")
            try:
                psblup_result = psblup(
                    Y=trait_subset,
                    X=snp_subset,
                    proximity_matrix=proximity_matrix,
                    log_transform_x=True,
                    percent_train_data=list(percent_train_data),
                    l2s=l2s,
                    runs=n_runs,
                )

                all_results[trait] = psblup_result[trait]
                psblup_success = True

                # Get psBLUP results
                accuracy_matrix = psblup_result[trait]["accuracy"]
                psblup_averages = np.mean(accuracy_matrix, axis=0)
                avg_best_psblup = np.max(psblup_averages)
                best_l2_idx = np.argmax(psblup_averages)
                best_l2 = l2s[best_l2_idx]

                print(
                    f"  ✓ psBLUP completed - Best average accuracy: {avg_best_psblup:.3f} (l2={best_l2})"
                )

            except Exception as e:
                print(f"  ✗ psBLUP Analysis failed: {e}")

            # Combined Results Summary
            if psblup_success or rrblup_success:
                print("\n  ====== COMBINED RESULTS SUMMARY ======")

                if rrblup_success:
                    print(f"  Average improved rrBLUP accuracy:     {avg_rrblup:.3f}")

                if psblup_success:
                    print(f"  Average Best psBLUP accuracy: {avg_best_psblup:.3f}")

                    if rrblup_success:
                        improvement = avg_best_psblup - avg_rrblup
                        print(f"  Overall improvement:         {improvement:+.3f}")

                    # Show psBLUP performance by l2 value
                    print("\n  psBLUP Performance by l2 value:")
                    for i, l2_val in enumerate(l2s):
                        l2_avg = psblup_averages[i]
                        if rrblup_success:
                            improvement = l2_avg - avg_rrblup
                            print(
                                f"    l2 = {l2_val:3d}: {l2_avg:.4f} (improvement: {improvement:+.4f})"
                            )
                        else:
                            print(f"    l2 = {l2_val:3d}: {l2_avg:.4f}")
        else:
            print("  ✗ Insufficient samples for analysis")

    return all_results, all_rrblup_results


def summarize_results(all_results: Dict, all_rrblup_results: Dict) -> None:
    """Generate final analysis summary"""

    print("\n" + "=" * 50)
    print("FINAL ANALYSIS SUMMARY")
    print("=" * 50)

    successful_traits = list(all_results.keys())
    successful_rrblup_traits = list(all_rrblup_results.keys())
    common_traits = list(set(successful_traits) & set(successful_rrblup_traits))

    print(f"Successfully analyzed traits with psBLUP: {len(successful_traits)}")
    print(
        f"Successfully analyzed traits with improved rrBLUP: {len(successful_rrblup_traits)}"
    )
    print(f"Traits analyzed by both methods: {len(common_traits)}")

    if len(successful_traits) > 0:
        print(f"psBLUP traits: {', '.join(successful_traits)}")
    if len(successful_rrblup_traits) > 0:
        print(f"Improved rrBLUP traits: {', '.join(successful_rrblup_traits)}")

    # Overall performance comparison
    if len(common_traits) > 0:
        print(f"\n{'='*50}")
        print("OVERALL PERFORMANCE COMPARISON")
        print("=" * 50)

        overall_improvements = []

        for trait in common_traits:
            # Get psBLUP best average accuracy
            psblup_acc_matrix = all_results[trait]["accuracy"]
            psblup_best_avg = np.max(np.mean(psblup_acc_matrix, axis=0))

            # Get rrBLUP average accuracy
            rrblup_acc_matrix = all_rrblup_results[trait]["accuracy"]
            rrblup_avg = np.mean(rrblup_acc_matrix[:, 0])

            # Calculate improvement
            improvement = psblup_best_avg - rrblup_avg
            overall_improvements.append(improvement)

            print(
                f"{trait:20s}: psBLUP={psblup_best_avg:.3f}, improved_rrBLUP={rrblup_avg:.3f}, "
                f"improvement={improvement:+.3f}"
            )

        overall_improvements = np.array(overall_improvements)
        mean_improvement = np.mean(overall_improvements)
        positive_improvements = np.sum(overall_improvements > 0)

        print(f"\nAverage improvement across all traits: {mean_improvement:+.4f}")
        print(
            f"Traits with positive improvement: {positive_improvements}/{len(common_traits)} "
            f"({100 * positive_improvements / len(common_traits):.1f}%)"
        )
        print(
            f"Range of improvements: {np.min(overall_improvements):.4f} to "
            f"{np.max(overall_improvements):.4f}"
        )

        # Save all results in CSV format
    if len(successful_traits) > 0 or len(successful_rrblup_traits) > 0:
        # Save detailed results in CSV format
        save_detailed_csv_results(all_results, all_rrblup_results)

    else:
        print("\nNo traits were successfully analyzed")


def save_detailed_csv_results(all_results: Dict, all_rrblup_results: Dict) -> None:
    """Save detailed analysis results in CSV format"""

    # Create output directory if it doesn't exist
    os.makedirs("./output_py", exist_ok=True)

    # Parameters used in analysis
    l2_values = np.arange(10, 190, 10)
    training_proportions = np.arange(0.3, 0.85, 0.1)

    # Save psBLUP detailed results
    if all_results:
        psblup_data = []
        for trait, result in all_results.items():
            accuracy_matrix = result["accuracy"]
            for i, train_prop in enumerate(training_proportions):
                for j, l2_val in enumerate(l2_values):
                    if i < accuracy_matrix.shape[0] and j < accuracy_matrix.shape[1]:
                        psblup_data.append(
                            {
                                "trait": trait,
                                "training_proportion": train_prop,
                                "l2_value": l2_val,
                                "accuracy": accuracy_matrix[i, j],
                            }
                        )

        psblup_df = pd.DataFrame(psblup_data)
        psblup_df.to_csv("./output_py/psBLUP_detailed_results_py.csv", index=False)
        print(
            "Detailed psBLUP results saved to './output_py/psBLUP_detailed_results_py.csv'"
        )

    # Save rrBLUP detailed results
    if all_rrblup_results:
        rrblup_data = []
        for trait, result in all_rrblup_results.items():
            accuracy_matrix = result["accuracy"]
            for i, train_prop in enumerate(training_proportions):
                if i < accuracy_matrix.shape[0]:
                    rrblup_data.append(
                        {
                            "trait": trait,
                            "training_proportion": train_prop,
                            "accuracy": accuracy_matrix[i, 0],
                        }
                    )

        rrblup_df = pd.DataFrame(rrblup_data)
        rrblup_df.to_csv("./output_py/rrBLUP_detailed_results_py.csv", index=False)
        print(
            "Detailed improved rrBLUP results saved to './output_py/rrBLUP_detailed_results_py.csv'"
        )


def load_csv_results() -> Tuple[Dict, Dict]:
    """Load analysis results from CSV files"""

    try:
        # Load detailed psBLUP results
        psblup_df = pd.read_csv("./output_py/psBLUP_detailed_results_py.csv")

        # Convert back to nested dictionary format
        psblup_results = {}
        l2_values = sorted(psblup_df["l2_value"].unique())
        training_props = sorted(psblup_df["training_proportion"].unique())

        for trait in psblup_df["trait"].unique():
            trait_data = psblup_df[psblup_df["trait"] == trait]

            # Create accuracy matrix
            accuracy_matrix = np.zeros((len(training_props), len(l2_values)))

            for i, train_prop in enumerate(training_props):
                for j, l2_val in enumerate(l2_values):
                    row = trait_data[
                        (trait_data["training_proportion"] == train_prop)
                        & (trait_data["l2_value"] == l2_val)
                    ]
                    if not row.empty:
                        accuracy_matrix[i, j] = row["accuracy"].iloc[0]

            psblup_results[trait] = {"accuracy": accuracy_matrix}

        # Load detailed rrBLUP results
        rrblup_df = pd.read_csv("./output_py/rrBLUP_detailed_results_py.csv")
        rrblup_results = {}

        for trait in rrblup_df["trait"].unique():
            trait_data = rrblup_df[rrblup_df["trait"] == trait]

            # Create accuracy matrix
            accuracy_matrix = np.zeros((len(training_props), 1))

            for i, train_prop in enumerate(training_props):
                row = trait_data[trait_data["training_proportion"] == train_prop]
                if not row.empty:
                    accuracy_matrix[i, 0] = row["accuracy"].iloc[0]

            rrblup_results[trait] = {"accuracy": accuracy_matrix}

        print("Results loaded from CSV files")
        return psblup_results, rrblup_results

    except FileNotFoundError as e:
        print(f"CSV files not found: {e}")
        return {}, {}
    except Exception as e:
        print(f"Error loading CSV results: {e}")
        return {}, {}


def main():
    """Main analysis function"""

    print("Starting psBLUP vs Improved rrBLUP Analysis")
    print("=" * 50)
    print("Using improved rrBLUP with REML estimation (better R compatibility)")
    print("-" * 50)

    try:
        # Load data
        met_data, snp_data, snp_map = load_data()

        # Quality control
        met_data_subset, snp_data_filtered, selected_traits = quality_control(
            met_data, snp_data
        )

        # Create proximity matrix
        proximity_matrix = create_proximity_matrix(snp_data_filtered, snp_map)

        # Run analysis
        all_results, all_rrblup_results = run_analysis(
            met_data_subset, snp_data_filtered, proximity_matrix, selected_traits
        )

        # Summarize results
        summarize_results(all_results, all_rrblup_results)

        print("\nAnalysis complete")

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
