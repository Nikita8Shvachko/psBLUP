# psBLUP Implementation

# Load required packages
packages <- c("data.table", "plyr", "progress")
lapply(packages, require, character.only = TRUE)

psBLUP <- function(
    Y = NULL,
    X = NULL,
    proximityMatrix = NULL,
    logTransformX = FALSE,
    percentTrainData = NULL,
    l2s = seq(from = 1, to = 75, length.out = 7),
    yVariables = NULL,
    xVariables = NULL,
    groupingVariables = NULL,
    runs = 5) {
    # Arguments validation
    if (is.null(X)) {
        stop("No predictor variables have been provided")
    }
    if (is.null(Y)) {
        stop("No dependent variables have been provided")
    }
    if (is.null(proximityMatrix)) {
        proximityMatrix <- cor(X)
    }
    if (!logTransformX %in% c(TRUE, FALSE)) {
        stop("Acceptable values for 'logTransformX': 'TRUE' or 'FALSE'")
    }
    if (is.null(yVariables)) {
        yVariables <- names(Y)
    }
    if (is.null(xVariables)) {
        xVariables <- names(X)
    }
    if (is.null(percentTrainData)) {
        percentTrainData <- 1
        runs <- 1
        warning(
            "Since no training data will be used, only 1 run will be conducted."
        )
    }

    objectsToReturn <- list()

    # Transform data to data.table format
    setDT(X, keep.rownames = TRUE)
    setDT(Y, keep.rownames = TRUE)

    # Keep only relevant variables
    Y <- Y[, c("rn", yVariables, groupingVariables), with = FALSE]
    X <- X[, c("rn", xVariables), with = FALSE]

    # Rename percentTrainData to proportion
    proportion <- percentTrainData

    # Apply log transformation to the data
    if (logTransformX) {
        Y[,
            c(yVariables) := lapply(.SD, function(x) log(x)),
            .SDcols = yVariables
        ]
    }

    # Handle grouping variables (remove grouping effect by regressing them out)
    if (!is.null(groupingVariables)) {
        meanByGroup <- copy(Y)
        f <- as.formula(paste(
            "x",
            paste(groupingVariables, collapse = "+"),
            sep = "~"
        ))
        meanByGroup[,
            c(yVariables) := lapply(.SD, function(x) {
                aov(formula = x ~ Treatment)$fitted.values
            }),
            .SDcols = yVariables
        ]
        Y[,
            c(yVariables) := lapply(.SD, function(x) {
                aov(formula = x ~ Treatment)$resid
            }),
            .SDcols = yVariables
        ]
        meanByGroup[,
            c(groupingVariables) := lapply(.SD, function(c) NULL),
            .SDcols = groupingVariables
        ]
        Y[,
            c(groupingVariables) := lapply(.SD, function(c) NULL),
            .SDcols = groupingVariables
        ]
    }

    # Add strength (row-sum) as diagonal (needed for properly defining the Laplacian matrix)
    diag(proximityMatrix) <- apply(proximityMatrix, 1, sum)

    # Scale the X variables
    X[,
        c(xVariables) := lapply(.SD, function(x) scale(x)),
        .SDcols = xVariables
    ]

    # Define N as the number of samples
    N <- nrow(X)

    # Run psBLUP for every metabolite
    cat("psBLUP: Processing")

    for (yVar in yVariables) {
        # Define response as scaled concentration of the selected metabolite
        response <- c(scale(Y[, yVar, with = FALSE]))

        # Define predictors as the SNP data
        predictors <- copy(X)

        # Find prediction variables that are negatively correlated to the response
        negativeCorrelations <- which(
            cor(response, predictors[, xVariables, with = FALSE]) < 0
        )
        neg.effects <- xVariables[negativeCorrelations]
        predictors[,
            c(neg.effects) := lapply(.SD, function(x) {
                a <- min(x)
                b <- max(x)
                return(mapvalues(x, from = c(a, b), to = c(b, a)))
            }),
            .SDcols = neg.effects
        ]

        # Define the Normalized Laplacian matrix as G
        G <- proximityMatrix
        diagG <- diag(G)
        G <- (-G)
        diag(G) <- diagG

        # Decompose the Normalized Laplacian matrix
        eigenvalues <- eigen(G)$values
        eigenvalues[which(eigenvalues < 0)] <- 0
        eigenvectors <- eigen(G)$vectors

        # Define newdata for the augmented data solution
        newdata <- t(eigenvectors %*% diag(sqrt(eigenvalues)))
        colnames(newdata) <- rownames(newdata) <- xVariables

        # Create matrix containing the psBLUP accuracy results
        overallAccuracy <- matrix(0, length(proportion), length(l2s))
        colnames(overallAccuracy) <- paste0("l2 = ", signif(l2s, digits = 3))
        rownames(overallAccuracy) <- paste0(
            "train = ",
            signif(proportion, digits = 2)
        )

        # Create matrix to store accuracy statistics per run
        accuracyStats <- matrix(NA, length(proportion), runs)
        colnames(accuracyStats) <- paste0("run: ", 1:runs)
        rownames(accuracyStats) <- paste0(
            "train sample: ",
            proportion * 100,
            "%"
        )

        # Run accuracy evaluation
        # progress bar for runs
        pb_runs <- progress_bar$new(
            format = "    Processing :what [:bar] :percent (:current/:total)",
            total = runs,
            clear = FALSE,
            width = 100
        )

        j <- 0
        while (j < runs) {
            # Create matrix for storing accuracy per sampling scenario
            accuracyPerScenario <- matrix(NA, length(proportion), length(l2s))
            colnames(accuracyPerScenario) <- paste0(
                "l2 = ",
                signif(l2s, digits = 3)
            )
            rownames(accuracyPerScenario) <- paste0(
                "train = ",
                signif(proportion, digits = 2)
            )

            # Advanced progress bar for training proportions
            pb_prop <- progress_bar$new(
                format = "      Train Props [:bar] :percent (:current/:total)",
                total = length(proportion),
                clear = TRUE,
                width = 100
            )

            for (k in 1:length(proportion)) {
                # Select training and testing data
                train <- sample(N, round(proportion[k] * N))
                test <- c(1:N)[-train]
                X.train <- predictors[train, ]
                Y.train <- response[train]

                if (length(train) == N) {
                    X.test <- X.train
                    Y.test <- Y.train
                } else {
                    X.test <- predictors[test, ]
                    Y.test <- response[test]
                }

                # Run psBLUP for every l2 value
                # Advanced progress bar for l2 values
                pb_l2 <- progress_bar$new(
                    format = "        L2 [:bar] :percent (:current/:total)",
                    total = length(l2s),
                    clear = TRUE,
                    width = 100
                )

                for (i in 1:length(l2s)) {
                    l2 <- l2s[i]

                    # Calculate factor for augmented data solution
                    factor <- (1 + l2)^(-1 / 2)

                    # Create augmented data for predictors and response
                    Xdata <- factor *
                        rbind(
                            X.train[, xVariables, with = FALSE],
                            sqrt(l2) * newdata
                        )
                    Ydata <- c(Y.train, rep(0, times = nrow(newdata)))
                    Xdata <- data.frame(Xdata)

                    # Fit psBLUP using basic linear regression (avoiding rrBLUP dependency)
                    mixed.new <- tryCatch(
                        {
                            lm.fit(as.matrix(Xdata), Ydata)
                        },
                        error = function(e) NA
                    )

                    # Check if model components have been defined correctly
                    if (any(is.na(mixed.new))) {
                        break
                    } else {
                        # Obtain psBLUP coefficients
                        coeffsPSBLUP <- mixed.new$coefficients

                        # Obtain fitted values
                        fittedPSBLUP <- as.matrix(X.test[,
                            xVariables,
                            with = FALSE
                        ]) %*%
                            coeffsPSBLUP

                        # Calculate psBLUP accuracy for specific l2
                        accuracyPerScenario[k, i] <- signif(
                            cor(Y.test, fittedPSBLUP),
                            digits = 3
                        )
                    }
                    pb_l2$tick()
                }

                # Store best accuracy for this run
                accuracyStats[k, j + 1] <- max(
                    accuracyPerScenario[k, ],
                    na.rm = TRUE
                )
                pb_prop$tick()
            }

            j <- j + 1
            pb_runs$tick(tokens = list(what = paste("Run", j)))
            overallAccuracy <- overallAccuracy + accuracyPerScenario
        }

        overallAccuracy <- overallAccuracy / j
        objectsToReturn[[yVar]] <- list(
            "accuracy" = overallAccuracy,
            "stats" = accuracyStats
        )
    }
    cat("\npsBLUP processing completed!\n")

    return(objectsToReturn)
}
