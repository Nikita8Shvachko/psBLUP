##### Final psBLUP Analysis with Proper Missing Data Handling ----
rm(list = ls())
path <- "/Users/askoritan/programming/psBLUP/psBLUP-main"
# Source the psBLUP function
source(paste0(path, "/psBLUP.R"))

# Load required packages
packages <- c("Matrix", "data.table", "plyr", "rrBLUP")
lapply(packages, require, character.only = TRUE)

##### Load Data ----
met.data <- read.csv(paste0(path, "/metabolites.csv"), header = TRUE, row.names = 1) # nolint: object_name_linter.
SNP.data <- read.csv(paste0(path, "/SNPs.csv"), header = TRUE, row.names = 1) # nolint
SNP.map <- read.csv(paste0(path, "/SNPmap.csv"), header = TRUE, row.names = 1)

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
min_samples <- 50
good_traits <- names(complete_per_trait)[complete_per_trait >= min_samples]
cat(
  "Traits with >=",
  min_samples,
  "complete samples:",
  length(good_traits),
  "\n"
)

if (length(good_traits) == 0) {
  cat("No traits with sufficient sample size. Lowering threshold to 50...\n")
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

# Select subset of traits for analysis (max 5 for demonstration)
selected_traits <- head(good_traits, 10)
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

##### Run psBLUP Analysis ----
cat("\n=============== Running psBLUP Analysis ===============\n")

# Create results storage
all_results <- list()

# Analyze each trait separately with proper sample filtering
for (trait in selected_traits) {
  # green color for the trait name
  cat(
    "\033[32m",
    "################################################",
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
    # Run psBLUP
    result <- try(
      {
        # Parameters
        percentTrainData <- c(0.3, 0.4, 0.5, 0.6, 0.7)
        l2s <- seq(10, 180, by = 20)
        runs <- 10

        psBLUP(
          Y = trait_subset,
          X = SNP_subset,
          proximityMatrix = proximityMatrix,
          logTransformX = TRUE,
          percentTrainData = percentTrainData,
          l2s = l2s,
          runs = runs,
          compareToRRBLUP = TRUE
        )
      },
      silent = FALSE
    )

    if (!inherits(result, "try-error")) {
      all_results[[trait]] <- result[[1]]
      cat("  ✓ Analysis completed successfully\n")

      # Print accuracy results correctly
      accuracy_matrix <- result[[1]]$accuracy
      gain_matrix <- result[[1]]$gain

      cat("  Detailed Accuracy Results:\n")
      print(accuracy_matrix)

      # Calculate overall averages across all training proportions
      rrblup_accuracies <- accuracy_matrix[, "RRBLUP"]
      avg_rrblup <- mean(rrblup_accuracies)

      # Find best psBLUP accuracy for each training proportion
      psblup_cols <- grep("l2 =", colnames(accuracy_matrix))
      best_psblup_accuracies <- apply(
        accuracy_matrix[, psblup_cols, drop = FALSE],
        1,
        max
      )
      avg_best_psblup <- mean(best_psblup_accuracies)

      # Calculate average for each l2 value
      psblup_averages <- apply(
        accuracy_matrix[, psblup_cols, drop = FALSE],
        2,
        mean
      )
      best_l2_overall <- names(psblup_averages)[which.max(
        psblup_averages
      )]

      cat(
        "\n  ====== SUMMARY AVERAGES (All Runs & Training Proportions) ======\n"
      )
      cat(sprintf("  Average RRBLUP accuracy:     %.3f\n||", avg_rrblup))
      cat(sprintf(
        "  Average Best psBLUP accuracy: %.3f\n",
        avg_best_psblup
      ))
      gain_acc <- result[[1]]$gain
      cat(sprintf(
        "  Overall improvement:         %.3f\n",
        mean(gain_acc)
      ))
      cat(sprintf(
        "  Best performing l2 value:    %s (avg=%.3f)\n",
        best_l2_overall,
        max(psblup_averages)
      ))

      cat(
        "\n  psBLUP Performance by l2 value (averaged across all scenarios):\n"
      )
      for (i in 1:length(psblup_averages)) {
        l2_name <- names(psblup_averages)[i]
        l2_avg <- psblup_averages[i]
        improvement <- l2_avg - avg_rrblup
        cat(sprintf(
          "    %s: %.4f (improvement: %+.4f)\n",
          l2_name,
          l2_avg,
          improvement
        ))
      }

      # Show gain statistics
      cat("\n  Accuracy Gain Statistics (psBLUP - RRBLUP per run):\n")
      cat(sprintf("    Mean gain:   %+.4f\n", mean(gain_matrix)))
      cat(sprintf("    Min gain:    %+.4f\n", min(gain_matrix)))
      cat(sprintf("    Max gain:    %+.4f\n", max(gain_matrix)))
      cat(sprintf(
        "    Positive gains: %d/%d runs (%.1f%%)\n",
        sum(gain_matrix > 0),
        length(gain_matrix),
        100 * sum(gain_matrix > 0) / length(gain_matrix)
      ))
    } else {
      cat(" Analysis failed:", as.character(result), "\n")
    }
  } else {
    cat(" Insufficient samples for analysis\n")
  }
}

##### Summary
cat("\n=== Analysis Summary ===\n")
successful_traits <- names(all_results)
cat("Successfully analyzed traits:", length(successful_traits), "\n")
if (length(successful_traits) > 0) {
  cat("Traits:", paste(successful_traits, collapse = ", "), "\n")

  # Save results
  save(all_results, file = "psBLUP_final_results.RData")
  cat("Results saved to 'psBLUP_final_results.RData'\n")
} else {
  cat("No traits were successfully analyzed\n")
}

cat("\n Analysis complete\n")
