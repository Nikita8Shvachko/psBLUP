##### Chickpea TSW psBLUP and rrBLUP Analysis comparison ----
rm(list = ls())
path_to_psBLUP <- "/Users/askoritan/programming/psBLUP"
# Source the psBLUP function
source(paste0(path_to_psBLUP, "/psBLUP_raw.R"))
# Source the rrBLUP function
source(paste0(path_to_psBLUP, "/rrBLUP.R"))

# Load packages
packages <- c("Matrix", "data.table", "plyr", "rrBLUP", "progress")
lapply(packages, require, character.only = TRUE)

##### Load Chickpea Data ----
path_to_data <- "/Users/askoritan/programming/psBLUP/nuts_data"

cat("Loading data...\n")

# Load phenotype data
pheno_data <- read.csv(
    paste0(path_to_data, "/pheno_2016_VIRVFVIR_421_408_synchro.csv"),
    header = TRUE,
    stringsAsFactors = FALSE
)

# Load genotype data
geno_data <- read.csv(
    paste0(path_to_data, "/total_df_for_aio_chickpea_28042016_synchro.csv"),
    header = TRUE,
    stringsAsFactors = FALSE,
    row.names = 1
)

cat(
    "Pheno:",
    nrow(pheno_data),
    "x",
    ncol(pheno_data),
    " | Geno:",
    nrow(geno_data),
    "x",
    ncol(geno_data),
    "\n"
)

##### Data Preparation (phenotype and genotype data) ----
# Extract TSW trait data
tsw_data <- pheno_data[, c("SNP.ID", "TSW")]
names(tsw_data) <- c("sample_id", "TSW")

# Remove rows with missing TSW values
tsw_data <- tsw_data[!is.na(tsw_data$TSW), ]

# Set sample IDs as rownames for TSW data
rownames(tsw_data) <- tsw_data$sample_id
tsw_data <- tsw_data[, "TSW", drop = FALSE]

# Ensure genotype data sample IDs match phenotype data sample IDs
common_samples <- intersect(rownames(tsw_data), rownames(geno_data))

cat("Common samples:", length(common_samples), "\n")

if (length(common_samples) == 0) {
    stop("No common samples found!")
}

# Filter to common samples
SNP.data <- geno_data[common_samples, ]
met.data <- tsw_data[common_samples, , drop = FALSE]
N <- length(common_samples)

cat(
    "Final: SNP",
    nrow(SNP.data),
    "x",
    ncol(SNP.data),
    " | TSW",
    nrow(met.data),
    "x",
    ncol(met.data),
    "\n"
)

##### Quality Control ----
# Remove SNPs with no variation
no_variation <- sapply(SNP.data, function(x) {
    unique_vals <- unique(x[!is.na(x)])
    length(unique_vals) <= 1
})

if (sum(no_variation) > 0) {
    cat("Removed", sum(no_variation), "invariant SNPs\n")
    SNP.data <- SNP.data[, !no_variation, drop = FALSE]
}

complete_samples <- sum(!is.na(met.data$TSW))
cat("Complete TSW samples:", complete_samples, "\n")

if (complete_samples < 30) {
    stop("Insufficient samples (need ≥30)")
}

##### Create Proximity Matrix ----
cat("Creating proximity matrix...\n")

# Reduce SNP set
max_snps <- 1000
if (ncol(SNP.data) > max_snps) {
    cat("Reducing SNPs:", ncol(SNP.data), "→", max_snps, "\n")
    selected_snps <- sample(colnames(SNP.data), max_snps)
    SNP.data_reduced <- SNP.data[, selected_snps, drop = FALSE]
} else {
    SNP.data_reduced <- SNP.data
}

# Calculate proximity matrix
cor_matrix <- cor(SNP.data_reduced, use = "complete.obs")
cor_matrix[is.na(cor_matrix)] <- 0
proximityMatrix <- cor_matrix^2
SNP.data <- SNP.data_reduced

cat(
    "Proximity matrix:",
    nrow(proximityMatrix),
    "x",
    ncol(proximityMatrix),
    "\n"
)

##### Run Combined psBLUP and rrBLUP Analysis ----
cat(
    "\n=============== Running psBLUP and rrBLUP Analysis for TSW ===============\n"
)

# Parameters
n_runs <- 10
l2s <- seq(1, 100, by = 7)
percentTrainDatas <- seq(0.2, 0.7, by = 0.1)

# Initialize
psblup_success <- FALSE
rrblup_success <- FALSE
avg_rrblup <- NA
avg_best_psblup <- NA
rrblup_result <- NULL
psblup_result <- NULL

cat("\nAnalyzing TSW trait with", complete_samples, "complete samples\n")

# Run rrBLUP first
cat("\n--- Running rrBLUP ---\n")

pb_rrblup <- progress_bar$new(
    format = "  rrBLUP [:bar] :percent ETA: :eta",
    total = 1,
    clear = FALSE,
    width = 100
)

rrblup_result <- try(
    {
        result <- rrBLUP(
            Y = met.data,
            X = SNP.data,
            logTransformX = TRUE,
            percentTrainData = percentTrainDatas,
            runs = n_runs
        )
        pb_rrblup$tick()
        result
    },
    silent = FALSE
)
cat("\nrrBLUP completed\n")
if (!inherits(rrblup_result, "try-error")) {
    rrblup_success <- TRUE

    # Get rrBLUP results
    rrblup_accuracy_matrix <- rrblup_result[[1]]$accuracy
    avg_rrblup <- mean(rrblup_accuracy_matrix[, "RRBLUP"])
    cat(sprintf(
        " rrBLUP completed - Average accuracy: %.3f\n",
        avg_rrblup
    ))
} else {
    cat("rrBLUP Analysis failed:", as.character(rrblup_result), "\n")
}

# Run psBLUP
cat("\n--- Running psBLUP ---\n")
# Advanced progress bar for psBLUP
pb_psblup <- progress_bar$new(
    format = "  psBLUP [:bar] :percent ETA: :eta",
    total = 1,
    clear = FALSE,
    width = 100
)

psblup_result <- try(
    {
        result <- psBLUP(
            Y = met.data,
            X = SNP.data,
            proximityMatrix = proximityMatrix,
            logTransformX = TRUE,
            percentTrainData = percentTrainDatas,
            l2s = l2s,
            runs = n_runs
        )
        pb_psblup$tick()
        result
    },
    silent = FALSE
)
cat("\npsBLUP completed\n")

if (!inherits(psblup_result, "try-error")) {
    psblup_success <- TRUE

    # Get psBLUP results
    accuracy_matrix <- psblup_result[[1]]$accuracy
    psblup_averages <- apply(accuracy_matrix, 2, mean)
    avg_best_psblup <- max(psblup_averages)
    best_l2_overall <- names(psblup_averages)[which.max(psblup_averages)]

    cat(sprintf(
        " psBLUP completed - Best average accuracy: %.3f (l2=%s)\n",
        avg_best_psblup,
        gsub("l2 = ", "", best_l2_overall)
    ))
} else {
    cat("psBLUP Analysis failed:", as.character(psblup_result), "\n")
}

##### Results ----
cat("\n=== RESULTS ===\n")

if (psblup_success || rrblup_success) {
    if (rrblup_success) {
        cat("rrBLUP:  ", sprintf("%.3f", avg_rrblup), "\n")
    }

    if (psblup_success) {
        cat("psBLUP:  ", sprintf("%.3f", avg_best_psblup), "\n")

        if (rrblup_success) {
            improvement <- avg_best_psblup - avg_rrblup
            cat(sprintf("Overall improvement:          %+.3f\n", improvement))

            if (improvement > 0) {
                cat("psBLUP shows improvement over rrBLUP\n")
            } else {
                cat("psBLUP does not improve over rrBLUP\n")
            }
        }

        # Show psBLUP performance by l2 value
        cat("\npsBLUP Performance by l2 value:\n")
        for (i in seq_len(length(psblup_averages))) {
            l2_name <- names(psblup_averages)[i]
            l2_avg <- psblup_averages[i]
            if (rrblup_success) {
                improvement <- l2_avg - avg_rrblup
                cat(sprintf(
                    "  %s: %.4f (improvement: %+.4f)\n",
                    l2_name,
                    l2_avg,
                    improvement
                ))
            } else {
                cat(sprintf("  %s: %.4f\n", l2_name, l2_avg))
            }
        }
    }

    cat("\n")
} else {
    cat("Both analyses failed\n")
}

##### Save Results ----
if (psblup_success || rrblup_success) {
    # Create output directory if it doesn't exist
    if (!dir.exists("./output_chickpea")) {
        dir.create("./output_chickpea")
    }

    # Save R data
    save(
        psblup_result,
        rrblup_result,
        file = "./output_chickpea/chickpea_tsw_results.RData"
    )
    cat("Results saved to 'output_chickpea/chickpea_tsw_results.RData'\n")

    # Save detailed CSV results
    cat("Saving detailed CSV results...\n")

    # Save detailed psBLUP results
    if (psblup_success) {
        psblup_detailed_data <- data.frame(
            trait = character(),
            training_proportion = numeric(),
            l2_value = numeric(),
            accuracy = numeric(),
            stringsAsFactors = FALSE
        )

        accuracy_matrix <- psblup_result[[1]]$accuracy

        for (i in seq_len(nrow(accuracy_matrix))) {
            train_prop <- percentTrainDatas[i]
            for (j in seq_len(ncol(accuracy_matrix))) {
                l2_val <- l2s[j]
                accuracy_val <- accuracy_matrix[i, j]

                psblup_detailed_data <- rbind(
                    psblup_detailed_data,
                    data.frame(
                        trait = "TSW",
                        training_proportion = train_prop,
                        l2_value = l2_val,
                        accuracy = accuracy_val,
                        stringsAsFactors = FALSE
                    )
                )
            }
        }

        write.csv(
            psblup_detailed_data,
            "./output_chickpea/chickpea_psBLUP_detailed_results.csv",
            row.names = FALSE
        )
        cat(
            "Detailed psBLUP results saved to 'chickpea_psBLUP_detailed_results.csv'\n"
        )
    }

    # Save detailed rrBLUP results
    if (rrblup_success) {
        rrblup_detailed_data <- data.frame(
            trait = character(),
            training_proportion = numeric(),
            accuracy = numeric(),
            stringsAsFactors = FALSE
        )

        accuracy_matrix <- rrblup_result[[1]]$accuracy

        for (i in seq_len(nrow(accuracy_matrix))) {
            train_prop <- percentTrainDatas[i]
            accuracy_val <- accuracy_matrix[i, 1] # rrBLUP has only 1 column

            rrblup_detailed_data <- rbind(
                rrblup_detailed_data,
                data.frame(
                    trait = "TSW",
                    training_proportion = train_prop,
                    accuracy = accuracy_val,
                    stringsAsFactors = FALSE
                )
            )
        }

        write.csv(
            rrblup_detailed_data,
            "./output_chickpea/chickpea_rrBLUP_detailed_results.csv",
            row.names = FALSE
        )
        cat(
            "Detailed rrBLUP results saved to 'chickpea_rrBLUP_detailed_results.csv'\n"
        )
    }
}

cat("Analysis complete\n")
