##### General options ----
# Clean objects in memory
rm(list = ls())

### If necessary, change the working directory to the one containing the data
workingDir <- "./"
setwd(workingDir)
options(stringsAsFactors = FALSE)

### Source the psBLUP function
source("./psBLUP.R")

### Packages that will be used for this analysis
packages <- c("Matrix")
lapply(packages, require, character.only = TRUE)

##### Load Data ----
### Load Metabolite data
met.data <- read.csv("./metabolites.csv", header = T, row.names = 1)
### Load SNP data and map of chromosomes
SNP.data <- read.csv("./SNPs.csv", header = T, row.names = 1)

### Find the common samples between SNP and metabolite data and sort the data sets in the same way
samples <- intersect(rownames(met.data), rownames(SNP.data))
SNP.data <- SNP.data[samples, ]
met.data <- met.data[samples, ]
N <- length(samples)

print(paste("Common samples:", N))
print(paste(
  "Original SNP data dimensions:",
  nrow(SNP.data),
  "x",
  ncol(SNP.data)
))

#### Data Quality Check - Remove SNPs with no variation ----
### Check for SNPs with no variation in the common samples
no_variation <- sapply(SNP.data, function(x) {
  unique_vals <- unique(x[!is.na(x)])
  length(unique_vals) <= 1
})

if (sum(no_variation) > 0) {
  print(paste(
    "Removing",
    sum(no_variation),
    "SNPs with no variation in common samples"
  ))
  SNP.data <- SNP.data[, !no_variation, drop = FALSE]
}

print(paste("Final SNP data dimensions:", nrow(SNP.data), "x", ncol(SNP.data)))

#### Reprocessing applied here ----
### Read chromosome map
SNP.map <- read.csv("./SNPmap.csv", header = T, row.names = 1)

### Keep only SNPs that remain after quality filtering
SNP.map <- SNP.map[colnames(SNP.data), ]

### Order by chromosome and position
SNP.map <- SNP.map[order(SNP.map$chromosome, SNP.map$position), ]
### Order SNP data columns by the order appearing in the SNP map
SNP.data <- SNP.data[, rownames(SNP.map)]
### Find number of chromosomes
n.chromosomes <- length(unique(SNP.map$chromosome))
### Create a list with each element being each chromosome containing marker information
chrom_list <- list()
for (i in 1:n.chromosomes) {
  SNPs <- rownames(SNP.map[SNP.map$chromosome == i, ])
  chrom_list[[i]] <- SNP.data[, SNPs]
}
### Make similarity matrix per chromosome
### The position of each marker is obtained
### Calculate the distance between each pair of markers
### For markers with distance less than 10cm, calculate the correlation matrix
### That is the similarity matrix per chromosome
matrices <- list()
for (i in 1:n.chromosomes) {
  chromosome <- chrom_list[[i]]
  positions <- SNP.map[SNP.map$chromosome == i, ]$position
  mat <- abs(outer(positions, positions, "-"))
  rownames(mat) <- colnames(mat) <- colnames(chromosome)

  # Calculate correlation matrix with proper handling of missing values
  cor_matrix <- cor(chromosome, use = "complete.obs")

  # Check for any remaining NA values and replace with 0
  cor_matrix[is.na(cor_matrix)] <- 0

  matrices[[i]] <- (cor_matrix^2) * (mat < 10)
}
### Combine the similarity matrices per chromosome to a block diagonal matrix
### with similarity 0 for markers belonging to different chromosomes
proximityMatrix <- as.matrix(bdiag(matrices))
colnames(proximityMatrix) <- rownames(proximityMatrix) <- colnames(SNP.data)

print("Proximity matrix created successfully")
print(paste(
  "Proximity matrix dimensions:",
  nrow(proximityMatrix),
  "x",
  ncol(proximityMatrix)
))

### Call psBLUP function with proper cross-validation
print("\n=== Running psBLUP Analysis ===")
print("Using cross-validation with multiple training proportions...")

results <- psBLUP(
  Y = met.data,
  X = SNP.data,
  proximityMatrix = proximityMatrix,
  logTransformX = FALSE,
  percentTrainData = c(0.4, 0.5, 0.6), # Multiple training proportions
  l2s = seq(from = 1, to = 75, length.out = 5),
  groupingVariables = NULL,
  runs = 3,
  compareToRRBLUP = TRUE
)

print("\n=== Analysis Complete ===")
print("Results saved in 'results' object")

# Save results to file
save(results, file = "psBLUP_results.RData")
print("Results saved to 'psBLUP_results.RData'")

# Print summary of first few traits
print("\n=== Sample Results ===")
trait_names <- names(results)[1:min(3, length(results))]
for (trait in trait_names) {
  cat("\nTrait:", trait, "\n")
  print(results[[trait]]$accuracy)
}
