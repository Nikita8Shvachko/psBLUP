import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def mixed_solve_approximation(
    y: np.ndarray, Z: np.ndarray, method: str = "REML"
) -> Dict:
    """
    Approximation of R's mixed.solve() function using REML estimation

    Parameters:
    -----------
    y : np.ndarray
        Response vector
    Z : np.ndarray
        Design matrix for random effects
    method : str
        Estimation method ("REML" or "ML")

    Returns:
    --------
    Dict with:
        - u: Random effect solutions (BLUP)
        - beta: Fixed effect solution
        - Ve: Error variance
        - Vu: Genetic variance (λ)
    """

    n = len(y)
    p = Z.shape[1]

    # Center the response
    y_mean = np.mean(y)
    y_centered = y - y_mean

    # Function to compute log-likelihood for variance component estimation
    def neg_log_likelihood(log_lambda):
        lambda_val = np.exp(log_lambda)  # λ = σ²e/σ²g

        # Mixed model equations: (Z'Z + λI)u = Z'y
        ZtZ = Z.T @ Z
        Zty = Z.T @ y_centered

        # Regularized system
        A = ZtZ + lambda_val * np.eye(p)

        try:
            # Solve for random effects
            u = solve(A, Zty, assume_a="pos")

            # Compute residuals
            residuals = y_centered - Z @ u

            # Log-likelihood
            sse = np.sum(residuals**2)
            log_det_A = np.linalg.slogdet(A)[1]

            if method == "REML":
                # REML log-likelihood
                ll = -0.5 * (n * np.log(sse) + log_det_A)
            else:
                # ML log-likelihood
                ll = -0.5 * (n * np.log(sse) + n * np.log(2 * np.pi))

            return -ll  # Return negative for minimization

        except np.linalg.LinAlgError:
            return np.inf

    # Optimize variance ratio
    try:
        result = minimize_scalar(neg_log_likelihood, bounds=(-10, 10), method="bounded")
        optimal_lambda = np.exp(result.x)
    except (RuntimeError, ValueError):
        # Fallback to default
        optimal_lambda = 1.0

    # Final solve with optimal lambda
    ZtZ = Z.T @ Z
    Zty = Z.T @ y_centered
    A = ZtZ + optimal_lambda * np.eye(p)

    try:
        u = solve(A, Zty, assume_a="pos")
    except np.linalg.LinAlgError:
        u = np.zeros(p)

    # Estimate variance components
    residuals = y_centered - Z @ u
    sse = np.sum(residuals**2)

    # Approximate variance components
    Ve = sse / (n - p) if n > p else 1.0  # Error variance
    Vu = Ve / optimal_lambda if optimal_lambda > 0 else 1.0  # Genetic variance

    return {
        "u": u,
        "beta": y_mean,
        "Ve": Ve,
        "Vu": Vu,
        "lambda": optimal_lambda,
    }


def improved_rrblup(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    log_transform_x: bool = False,
    percent_train_data: Optional[List[float]] = None,
    y_variables: Optional[List[str]] = None,
    x_variables: Optional[List[str]] = None,
    grouping_variables: Optional[List[str]] = None,
    runs: int = 5,
    method: str = "REML",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Improved rrBLUP implementation using mixed model approximation

    More faithful to R's rrBLUP package behavior than simple Ridge regression
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

    # Data preprocessing
    X_work = X.copy()
    Y_work = Y.copy()

    if grouping_variables:
        Y_work = Y_work[y_variables + grouping_variables]
    else:
        Y_work = Y_work[y_variables]
    X_work = X_work[x_variables]

    if log_transform_x:
        for var in y_variables:
            Y_work[var] = np.log(Y_work[var])

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

    # Run improved rrBLUP for every response variable
    for y_var in tqdm(y_variables, desc="Processing improved rrBLUP Y variables"):
        print(f"  Processing improved rrBLUP for variable: {y_var}")

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

        # Run cross-validation
        j = 0
        run_progress = tqdm(total=runs, desc=f"rrBLUP runs for {y_var}", leave=False)
        while j < runs:
            accuracy_per_scenario = np.full((len(percent_train_data), 1), np.nan)

            for k, proportion in enumerate(
                tqdm(percent_train_data, desc="Train proportions", leave=False)
            ):
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

                    # Fit improved rrBLUP using mixed model approximation
                    Z_train = X_train[x_variables].values
                    mixed_result = mixed_solve_approximation(
                        Y_train, Z_train, method=method
                    )

                    # Get predictions
                    Z_test = X_test[x_variables].values
                    Y_pred = Z_test @ mixed_result["u"] + mixed_result["beta"]

                    # Calculate accuracy
                    correlation, _ = pearsonr(Y_test, Y_pred)
                    accuracy_per_scenario[k, 0] = np.round(correlation, 3)

                except Exception as e:
                    print(
                        f"    Error in improved rrBLUP run {j+1}, proportion {k+1}: {e}"
                    )
                    break

                # Store accuracy for this run
                accuracy_stats[k, j] = accuracy_per_scenario[k, 0]

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
