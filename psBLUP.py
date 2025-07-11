import warnings
from typing import Dict, List, Optional

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def psblup(
    Y: Optional[pd.DataFrame] = None,
    X: Optional[pd.DataFrame] = None,
    proximity_matrix: Optional[np.ndarray] = None,
    log_transform_x: bool = False,
    percent_train_data: Optional[float] = None,
    l2s: np.ndarray = np.linspace(1, 75, 7),
    y_variables: Optional[List[str]] = None,
    x_variables: Optional[List[str]] = None,
    grouping_variables: Optional[List[str]] = None,
    runs: int = 5,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Penalized structured Best Linear Unbiased Prediction (psBLUP) method

    Parameters:
    -----------
    Y : pd.DataFrame
        Dependent variables (response matrix)
    X : pd.DataFrame
        Predictor variables (feature matrix)
    proximity_matrix : np.ndarray, optional
        Proximity matrix for structured penalty (default: correlation matrix of X)
    log_transform_x : bool, default False
        Whether to apply log transformation to Y variables
    percent_train_data : float, optional
        Proportion of data to use for training (default: 1.0, no train/test split)
    l2s : np.ndarray
        Array of L2 penalty parameters to test
    y_variables : List[str], optional
        Names of Y variables to analyze (default: all columns in Y)
    x_variables : List[str], optional
        Names of X variables to use (default: all columns in X)
    grouping_variables : List[str], optional
        Variables to remove grouping effects from
    runs : int, default 5
        Number of cross-validation runs

    Returns:
    --------
    Dict[str, Dict[str, np.ndarray]]
        Dictionary with results for each Y variable containing accuracy matrices
    """

    # Arguments validation
    if X is None:
        raise ValueError("No predictor variables have been provided")
    if Y is None:
        raise ValueError("No dependent variables have been provided")

    if proximity_matrix is None:
        proximity_matrix = np.corrcoef(X.select_dtypes(include=[np.number]).T)

    if not isinstance(log_transform_x, bool):
        raise ValueError("Acceptable values for 'log_transform_x': True or False")

    if y_variables is None:
        y_variables = list(Y.select_dtypes(include=[np.number]).columns)

    if x_variables is None:
        x_variables = list(X.select_dtypes(include=[np.number]).columns)

    if percent_train_data is None:
        percent_train_data = 1.0
        runs = 1
        warnings.warn(
            "Since no training data will be used, only 1 run will be conducted."
        )

    objects_to_return = {}

    # Create copies of the dataframes to avoid modifying originals
    X_work = X.copy()
    Y_work = Y.copy()

    # Keep only relevant variables
    if grouping_variables:
        Y_work = Y_work[y_variables + grouping_variables]
    else:
        Y_work = Y_work[y_variables]
    X_work = X_work[x_variables]

    # Rename percentTrainData to proportion
    proportion = (
        [percent_train_data]
        if isinstance(percent_train_data, (int, float))
        else percent_train_data
    )

    # Apply log transformation to the data
    if log_transform_x:
        for var in y_variables:
            Y_work[var] = np.log(Y_work[var])

    # Handle grouping variables
    if grouping_variables:
        Y_work = Y_work[y_variables]

    # Add strength (row-sum) as diagonal (needed for properly defining the Laplacian matrix)
    row_sums = np.sum(proximity_matrix, axis=1)
    np.fill_diagonal(proximity_matrix, row_sums)

    # Scale the X variables
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_work), columns=X_work.columns, index=X_work.index
    )

    # Define N as the number of samples
    N = len(X_scaled)

    # Run psBLUP for every metabolite/response variable
    for y_var in tqdm(y_variables, desc="Processing Y variables"):
        print(f"Processing variable: {y_var}")

        # Define response as scaled concentration of the selected variable
        y_scaler = StandardScaler()
        response = y_scaler.fit_transform(Y_work[[y_var]]).flatten()

        # Define predictors as the scaled data
        predictors = X_scaled.copy()

        # Find prediction variables that are negatively correlated to the response
        correlations = []
        for col in x_variables:
            corr, _ = pearsonr(response, predictors[col])
            correlations.append(corr)

        correlations = np.array(correlations)
        negative_correlations = np.where(correlations < 0)[0]
        neg_effects = [x_variables[i] for i in negative_correlations]

        # Flip negatively correlated variables
        for col in neg_effects:
            col_data = predictors[col]
            min_val, max_val = col_data.min(), col_data.max()
            # Map min to max and max to min
            predictors[col] = max_val + min_val - col_data

        # Define the Normalized Laplacian matrix as G
        G = proximity_matrix.copy()
        diag_G = np.diag(G).copy()
        G = -G
        np.fill_diagonal(G, diag_G)

        # Decompose the Normalized Laplacian matrix
        eigenvalues, eigenvectors = linalg.eigh(G)
        eigenvalues[eigenvalues < 0] = 0

        # Define newdata for the augmented data solution
        newdata = (eigenvectors @ np.diag(np.sqrt(eigenvalues))).T

        # Create matrix containing the psBLUP accuracy results
        overall_accuracy = np.zeros((len(proportion), len(l2s)))

        # Create matrix to store accuracy statistics per run
        accuracy_stats = np.full((len(proportion), runs), np.nan)

        # Run accuracy evaluation
        j = 0
        run_progress = tqdm(total=runs, desc=f"Runs for {y_var}", leave=False)
        while j < runs:
            # Create matrix for storing accuracy per sampling scenario
            accuracy_per_scenario = np.full((len(proportion), len(l2s)), np.nan)

            for k in tqdm(
                range(len(proportion)), desc="Train proportions", leave=False
            ):
                # Select training and testing data
                train_indices = np.random.choice(
                    N, size=int(np.round(proportion[k] * N)), replace=False
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

                # Run psBLUP for every l2 value
                for i, l2 in enumerate(tqdm(l2s, desc="L2 values", leave=False)):
                    try:
                        # Calculate factor for augmented data solution
                        factor = (1 + l2) ** (-1 / 2)

                        # Create augmented data for predictors and response
                        X_train_matrix = X_train[x_variables].values
                        augmented_X = factor * np.vstack(
                            [X_train_matrix, np.sqrt(l2) * newdata]
                        )
                        augmented_Y = np.concatenate(
                            [Y_train, np.zeros(newdata.shape[0])]
                        )

                        # Fit psBLUP using linear regression
                        model = LinearRegression(fit_intercept=False)
                        model.fit(augmented_X, augmented_Y)

                        # Obtain psBLUP coefficients
                        coeffs_psblup = model.coef_

                        # Obtain fitted values
                        X_test_matrix = X_test[x_variables].values
                        fitted_psblup = X_test_matrix @ coeffs_psblup

                        # Calculate psBLUP accuracy for specific l2
                        correlation, _ = pearsonr(Y_test, fitted_psblup)
                        accuracy_per_scenario[k, i] = np.round(correlation, 3)

                    except Exception as e:
                        print(f"Error in run {j+1}, proportion {k+1}, l2 {i+1}: {e}")
                        break

                # Store best accuracy for this run
                valid_accuracies = accuracy_per_scenario[
                    k, ~np.isnan(accuracy_per_scenario[k, :])
                ]
                if len(valid_accuracies) > 0:
                    accuracy_stats[k, j] = np.max(valid_accuracies)

            j += 1
            run_progress.update(1)
            overall_accuracy += np.nan_to_num(accuracy_per_scenario, nan=0)

        run_progress.close()
        overall_accuracy = overall_accuracy / j

        objects_to_return[y_var] = {
            "accuracy": overall_accuracy,
            "stats": accuracy_stats,
        }

    return objects_to_return


if __name__ == "__main__":
    pass
