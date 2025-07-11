##### Final psBLUP Analysis with Proper Missing Data Handling ----
rm(list = ls())
path_to_psBLUP <- "/Users/askoritan/programming/psBLUP"
# Source the psBLUP function
source(paste0(path_to_psBLUP, "/psBLUP_raw.R"))
# Source the rrBLUP function
source(paste0(path_to_psBLUP, "/rrBLUP.R"))

# Load required packages
packages <- c("Matrix", "data.table", "plyr", "rrBLUP")
lapply(packages, require, character.only = TRUE)
path_to_data <- "/Users/askoritan/programming/psBLUP/psBLUP-main"
##### Load Data ----
met.data <- read.csv(
  paste0(path_to_data, "/metabolites.csv"),
  header = TRUE,
  row.names = 1
) # nolint: object_name_linter.
SNP.data <- read.csv(
  paste0(path_to_data, "/SNPs.csv"),
  header = TRUE,
  row.names = 1
) # nolint
SNP.map <- read.csv(
  paste0(path_to_data, "/SNPmap.csv"),
  header = TRUE,
  row.names = 1
)

# Find common samples
samples <- intersect(rownames(met.data), rownames(SNP.data))
SNP.data <- SNP.data[samples, ]
met.data <- met.data[samples, ]
N <- length(samples)

cat("=============== Data Summary ===============\n")
cat("Common samples:", N, "\n")
cat("SNP data dimensions:", nrow(SNP.data), "x", ncol(SNP.data), "\n")
cat("Metabolite data dimensions:", nrow(met.data), "x", ncol(met.data), "\n")
cat("SNP missing values:", sum(is.na(SNP.data)), "\n")
cat("Metabolite missing values:", sum(is.na(met.data)), "\n")

##### Quality Control ----
# Remove SNPs with no variation
no_variation <- sapply(SNP.data, function(x) {
  unique_vals <- unique(x[!is.na(x)])
  length(unique_vals) <= 1
})

if (sum(no_variation) > 0) {
  cat("Removing", sum(no_variation), "SNPs with no variation\n")
  SNP.data <- SNP.data[, !no_variation, drop = FALSE]
}

# Filter metabolites to those with sufficient sample sizes
complete_per_trait <- sapply(met.data, function(x) sum(!is.na(x)))
cat("\nMetabolite completeness:\n")
cat("Min complete samples per trait:", min(complete_per_trait), "\n")
cat("Max complete samples per trait:", max(complete_per_trait), "\n")

# Select traits with at least 80 complete samples for robust analysis
min_samples <- 40
good_traits <- names(complete_per_trait)[complete_per_trait >= min_samples]
cat(
  "Traits with >=",
  min_samples,
  "complete samples:",
  length(good_traits),
  "\n"
)

if (length(good_traits) == 0) {
  cat("No traits with sufficient sample size. Lowering threshold to 40...\n")
  min_samples <- 40
  good_traits <- names(complete_per_trait)[complete_per_trait >= min_samples]
  cat(
    "Traits with >=",
    min_samples,
    "complete samples:",
    length(good_traits),
    "\n"
  )
}

if (length(good_traits) == 0) {
  stop("No traits have sufficient sample size for analysis") # nolint: indentation_linter.
}

# Select subset of traits for analysis (max 20 for demonstration)
selected_traits <- head(good_traits, 5)
met.data.subset <- met.data[, selected_traits, drop = FALSE]

cat("Selected traits for analysis:\n")
for (trait in selected_traits) {
  complete_count <- sum(!is.na(met.data.subset[, trait]))
  cat("  ", trait, ":", complete_count, "complete samples\n")
}

##### Create Proximity Matrix ----
# Keep only SNPs that remain after quality filtering
SNP.map <- SNP.map[colnames(SNP.data), ]

# Order by chromosome and position
SNP.map <- SNP.map[order(SNP.map$chromosome, SNP.map$position), ]
SNP.data <- SNP.data[, rownames(SNP.map)]
n.chromosomes <- length(unique(SNP.map$chromosome))

chrom_list <- list()
for (i in 1:n.chromosomes) {
  SNPs <- rownames(SNP.map[SNP.map$chromosome == i, ])
  chrom_list[[i]] <- SNP.data[, SNPs, drop = FALSE]
}

matrices <- list()
for (i in 1:n.chromosomes) {
  chromosome <- chrom_list[[i]]
  positions <- SNP.map[SNP.map$chromosome == i, ]$position
  mat <- abs(outer(positions, positions, "-"))
  rownames(mat) <- colnames(mat) <- colnames(chromosome)

  cor_matrix <- cor(chromosome, use = "complete.obs")
  cor_matrix[is.na(cor_matrix)] <- 0

  matrices[[i]] <- (cor_matrix^2) * (mat < 10)
}

proximityMatrix <- as.matrix(bdiag(matrices))
colnames(proximityMatrix) <- rownames(proximityMatrix) <- colnames(SNP.data)

cat(
  "\nProximity matrix created:",
  nrow(proximityMatrix),
  "x",
  ncol(proximityMatrix),
  "\n"
)

##### Run Combined psBLUP and rrBLUP Analysis ----
cat("\n=============== Running psBLUP and rrBLUP Analysis ===============\n")

# Create results storage
all_results <- list()
all_rrblup_results <- list()

# Analyze each trait separately with proper sample filtering
for (trait in selected_traits) {
  # green color for the trait name
  cat(
    "\033[32m",
    "|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|",
    "\033[0m"
  )
  cat("\nAnalyzing trait:", trait, "\n")

  # Get complete samples for this trait
  trait_data <- met.data.subset[, trait, drop = FALSE]
  complete_samples <- !is.na(trait_data[, 1])

  trait_subset <- trait_data[complete_samples, , drop = FALSE]
  SNP_subset <- SNP.data[complete_samples, ]

  n_complete <- nrow(trait_subset)
  cat("  Complete samples:", n_complete, "\n")

  if (n_complete >= 30) {
    # Variables to store results for comparison
    psblup_success <- FALSE
    rrblup_success <- FALSE
    avg_rrblup <- NA
    avg_best_psblup <- NA

    # Analysis parameters - synchronized with Python script
    n_runs <- 20
    l2s <- seq(10, 180, by = 10) # Same as Python: np.arange(10, 190, 10)
    percentTrainDatas <- seq(0.3, 0.8, by = 0.1) # Same as Python: np.arange(0.3, 0.85, 0.1)

    # Run rrBLUP first
    cat("\n  --- Running rrBLUP ---\n")
    rrblup_result <- try(
      {
        rrBLUP(
          Y = trait_subset,
          X = SNP_subset,
          logTransformX = TRUE,
          percentTrainData = percentTrainDatas,
          runs = n_runs
        )
      },
      silent = FALSE
    )

    if (!inherits(rrblup_result, "try-error")) {
      all_rrblup_results[[trait]] <- rrblup_result[[1]]
      rrblup_success <- TRUE

      # Get rrBLUP results
      rrblup_accuracy_matrix <- rrblup_result[[1]]$accuracy
      avg_rrblup <- mean(rrblup_accuracy_matrix[, "RRBLUP"])
      cat(sprintf(
        "  ✓ rrBLUP completed - Average accuracy: %.3f\n",
        avg_rrblup
      ))
    } else {
      cat("  ✗ rrBLUP Analysis failed:", as.character(rrblup_result), "\n")
    }

    # Run psBLUP
    cat("\n  --- Running psBLUP ---\n")
    result <- try(
      {
        psBLUP(
          Y = trait_subset,
          X = SNP_subset,
          proximityMatrix = proximityMatrix,
          logTransformX = TRUE,
          percentTrainData = percentTrainDatas,
          l2s = l2s,
          runs = n_runs
        )
      },
      silent = FALSE
    )

    if (!inherits(result, "try-error")) {
      all_results[[trait]] <- result[[1]]
      psblup_success <- TRUE

      # Get psBLUP results
      accuracy_matrix <- result[[1]]$accuracy
      psblup_averages <- apply(accuracy_matrix, 2, mean)
      avg_best_psblup <- max(psblup_averages)
      best_l2_overall <- names(psblup_averages)[which.max(psblup_averages)]

      cat(sprintf(
        "  ✓ psBLUP completed - Best average accuracy: %.3f (l2=%s)\n",
        avg_best_psblup,
        gsub("l2 = ", "", best_l2_overall)
      ))
    } else {
      cat("  ✗ psBLUP Analysis failed:", as.character(result), "\n")
    }

    # Combined Results Summary
    if (psblup_success || rrblup_success) {
      cat("\n  ====== COMBINED RESULTS SUMMARY ======\n")

      if (rrblup_success) {
        cat(sprintf("  Average rrBLUP accuracy:     %.3f\n", avg_rrblup))
      }

      if (psblup_success) {
        cat(sprintf("  Average Best psBLUP accuracy: %.3f\n", avg_best_psblup))

        if (rrblup_success) {
          improvement <- avg_best_psblup - avg_rrblup

          cat(sprintf(
            "  Overall improvement:         %+.3f\n",
            improvement
          ))
        }

        # Show psBLUP performance by l2 value
        cat("\n  psBLUP Performance by l2 value:\n")
        for (i in 1:length(psblup_averages)) {
          l2_name <- names(psblup_averages)[i]
          l2_avg <- psblup_averages[i]
          if (rrblup_success) {
            improvement <- l2_avg - avg_rrblup
            cat(sprintf(
              "    %s: %.4f (improvement: %+.4f)\n",
              l2_name,
              l2_avg,
              improvement
            ))
          } else {
            cat(sprintf("    %s: %.4f\n", l2_name, l2_avg))
          }
        }
      }

      cat("\n")
    }
  } else {
    cat("  ✗ Insufficient samples for analysis\n")
  }
}

##### Summary ----
cat("\n=== FINAL ANALYSIS SUMMARY ===\n")
successful_traits <- names(all_results)
successful_rrblup_traits <- names(all_rrblup_results)
common_traits <- intersect(successful_traits, successful_rrblup_traits)

cat(
  "Successfully analyzed traits with psBLUP:",
  length(successful_traits),
  "\n"
)
cat(
  "Successfully analyzed traits with rrBLUP:",
  length(successful_rrblup_traits),
  "\n"
)
cat("Traits analyzed by both methods:", length(common_traits), "\n")

if (length(successful_traits) > 0) {
  cat("psBLUP traits:", paste(successful_traits, collapse = ", "), "\n")
}
if (length(successful_rrblup_traits) > 0) {
  cat("rrBLUP traits:", paste(successful_rrblup_traits, collapse = ", "), "\n")
}

# Overall performance summary for traits analyzed by both methods
if (length(common_traits) > 0) {
  cat("\n=== OVERALL PERFORMANCE COMPARISON ===\n")
  overall_improvements <- numeric(length(common_traits))

  for (i in seq_along(common_traits)) {
    trait <- common_traits[i]

    # Get psBLUP best average accuracy
    psblup_acc_matrix <- all_results[[trait]]$accuracy
    psblup_best_avg <- max(apply(psblup_acc_matrix, 2, mean))

    # Get rrBLUP average accuracy
    rrblup_acc_matrix <- all_rrblup_results[[trait]]$accuracy
    rrblup_avg <- mean(rrblup_acc_matrix[, "RRBLUP"])

    # Calculate improvement
    improvement <- psblup_best_avg - rrblup_avg
    overall_improvements[i] <- improvement
  }

  mean_improvement <- mean(overall_improvements)
  positive_improvements <- sum(overall_improvements > 0)

  cat(sprintf(
    "Average improvement across all traits: %+.4f\n",
    mean_improvement
  ))
  cat(sprintf(
    "Traits with positive improvement: %d/%d (%.1f%%)\n",
    positive_improvements,
    length(common_traits),
    100 * positive_improvements / length(common_traits)
  ))
  cat(sprintf(
    "Range of improvements: %.4f to %.4f\n",
    min(overall_improvements),
    max(overall_improvements)
  ))
}


if (length(successful_traits) > 0 || length(successful_rrblup_traits) > 0) {
  # Save results in R format
  save(
    all_results,
    all_rrblup_results,
    file = "./output_R/psBLUP_final_results_test_raw_rr_ps.RData"
  )
  cat("Results saved to 'psBLUP_final_results_test_raw_rr_ps.RData'\n")

  # Save detailed CSV results (similar to Python version)
  cat("Saving detailed CSV results...\n")

  # Save detailed psBLUP results
  if (length(successful_traits) > 0) {
    psblup_detailed_data <- data.frame(
      trait = character(),
      training_proportion = numeric(),
      l2_value = numeric(),
      accuracy = numeric(),
      stringsAsFactors = FALSE
    )

    for (trait in successful_traits) {
      accuracy_matrix <- all_results[[trait]]$accuracy

      for (i in 1:nrow(accuracy_matrix)) {
        train_prop <- percentTrainDatas[i]
        for (j in 1:ncol(accuracy_matrix)) {
          l2_val <- l2s[j]
          accuracy_val <- accuracy_matrix[i, j]

          psblup_detailed_data <- rbind(
            psblup_detailed_data,
            data.frame(
              trait = trait,
              training_proportion = train_prop,
              l2_value = l2_val,
              accuracy = accuracy_val,
              stringsAsFactors = FALSE
            )
          )
        }
      }
    }

    write.csv(
      psblup_detailed_data,
      "./output_R/psBLUP_detailed_results_R.csv",
      row.names = FALSE
    )
    cat("Detailed psBLUP results saved to 'psBLUP_detailed_results_R.csv'\n")
  }

  # Save detailed rrBLUP results
  if (length(successful_rrblup_traits) > 0) {
    rrblup_detailed_data <- data.frame(
      trait = character(),
      training_proportion = numeric(),
      accuracy = numeric(),
      stringsAsFactors = FALSE
    )

    for (trait in successful_rrblup_traits) {
      accuracy_matrix <- all_rrblup_results[[trait]]$accuracy

      for (i in 1:nrow(accuracy_matrix)) {
        train_prop <- percentTrainDatas[i]
        accuracy_val <- accuracy_matrix[i, 1] # rrBLUP has only 1 column

        rrblup_detailed_data <- rbind(
          rrblup_detailed_data,
          data.frame(
            trait = trait,
            training_proportion = train_prop,
            accuracy = accuracy_val,
            stringsAsFactors = FALSE
          )
        )
      }
    }

    write.csv(
      rrblup_detailed_data,
      "./output_R/rrBLUP_detailed_results_R.csv",
      row.names = FALSE
    )
    cat("Detailed rrBLUP results saved to 'rrBLUP_detailed_results_R.csv'\n")
  }
} else {
  cat("No traits were successfully analyzed\n") # nolint
}

cat("\n✅ Analysis complete\n")
