### rrBLUP ----
rrBLUP <- function(
    Y = NULL,
    X = NULL,
    logTransformX = FALSE,
    percentTrainData = NULL,
    yVariables = NULL,
    xVariables = NULL,
    groupingVariables = NULL,
    runs = 5
) {
    ##############################################################################
    # Arguments:
    # Y: data.frame of the response variables
    # X: data.frame of the predictor variables
    # logTransformX: TRUE/FALSE on whether a log-transformation needs to be
    #                implemented on the response variables
    # percentTrainData: A vector with the percentages to be used for
    #                   training the model
    # yVariables: a vector of names for the response variables to be used
    # xVariables: a vector of names for the predictor variables to be used
    # groupingVariables: a vector of names for variable included in the predictor
    #                    variables that we need to remove their effect, e.g.
    #                    treatment effects
    # runs: number of runs for calculating the final accuracy
    ##############################################################################

    ### Load relevant packages
    packages <- c("data.table", "plyr", "rrBLUP", "progress")
    lapply(packages, require, character.only = TRUE)

    ### Do the necessary checks for arguments
    if (is.null(X)) {
        stop("No predictor variables have been provided")
    }
    if (is.null(Y)) {
        stop("No dependent variables have been provided")
    }
    if (!logTransformX %in% c(T, F)) {
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
        runs = 1
        warning(
            "Since no training data will be used, only 1 run will be conducted."
        )
    }
    objectsToReturn <- list()

    ### Convert data.frame to data.table format if needed
    X_dt <- copy(data.table(X, keep.rownames = TRUE))
    Y_dt <- copy(data.table(Y, keep.rownames = TRUE))

    ### Keep only relevant variables
    Y_dt <- Y_dt[, c("rn", yVariables, groupingVariables), with = F]
    ### Keep only relevant variables
    X_dt <- X_dt[, c("rn", xVariables), with = F]

    ### Rename percentTrainData to proportion
    proportion = percentTrainData

    ### Apply log transformation to the data
    if (logTransformX) {
        Y_dt[,
            c(yVariables) := lapply(.SD, function(x) {
                log(x)
            }),
            .SDcols = yVariables
        ]
    }

    ### Keep mean level of dependent variables per grouping variable
    ### Also remove grouping effect by regressing them out
    if (!is.null(groupingVariables)) {
        meanByGroup <- copy(Y_dt)
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
        Y_dt[,
            c(yVariables) := lapply(.SD, function(x) {
                aov(formula = x ~ Treatment)$resid
            }),
            .SDcols = yVariables
        ]
        meanByGroup[,
            c(groupingVariables) := lapply(.SD, function(c) {
                NULL
            }),
            .SDcols = groupingVariables
        ]
        Y_dt[,
            c(groupingVariables) := lapply(.SD, function(c) {
                NULL
            }),
            .SDcols = groupingVariables
        ]
    }

    ### Scale the X variables
    X_dt[,
        c(xVariables) := lapply(.SD, function(x) {
            scale(x)
        }),
        .SDcols = xVariables
    ]

    ### Define N as the number of samples
    N <- nrow(X_dt)

    ### Run rrBLUP for every metabolite
    # Advanced progress bar for Y variables
    cat("rrBLUP: Processing", length(yVariables), "response variable(s)\n")
    pb_yvar <- progress_bar$new(
        format = "  RRBLUP processing [:bar] :percent (:current/:total)",
        total = length(yVariables),
        clear = TRUE,
        width = 100
    )

    for (yVar in yVariables) {
        ### define as "response" the scaled concentration of the selected metabolite
        response <- c(scale(Y_dt[, yVar, with = F]))
        ### define as "predictors" the SNP data
        predictors <- copy(X_dt)

        ### Find which prediction variables that are negatively correlated to the response so that
        ### neighboring SNPs with opposing effects will not negate true signal
        negativeCorrelations <- which(
            cor(response, predictors[, xVariables, with = F]) < 0
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

        ### Create a matrix to store accuracy statistics per run
        accuracy <- matrix(NA, length(proportion), runs)
        colnames(accuracy) <- paste0("run: ", 1:runs)
        rownames(accuracy) <- paste0("train sample: ", proportion * 100, "%")

        ### Create a matrix containing the final accuracy results
        ### which will be the aggregated results over all runs
        overallAccuracy <- matrix(0, length(proportion), 1)
        colnames(overallAccuracy) <- "RRBLUP"
        rownames(overallAccuracy) <- paste0(
            "train = ",
            signif(proportion, digits = 2)
        )

        ### Run accuracy evaluation using while, as the samples might not contain genetic variation
        ### and produce error in fitting the models

        # Advanced progress bar for runs
        pb_runs <- progress_bar$new(
            format = "    Processing :what [:bar] :percent (:current/:total)",
            total = runs,
            clear = TRUE,
            width = 100
        )

        j <- 0
        while (j < runs) {
            ### Create a matrix for storing the accuracy per sampling scenario
            accuracyPerScenario <- matrix(NA, length(proportion), 1)
            colnames(accuracyPerScenario) <- "RRBLUP"
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
                ### Select the training and testing data
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

                ### Fit RRBLUP
                mixed.new <- tryCatch(
                    mixed.solve(
                        Y.train,
                        Z = X.train[, xVariables, with = F],
                        method = "REML"
                    ),
                    error = function(e) NA
                )

                ### Check if model components have been defined correctly
                if (any(is.na(mixed.new))) {
                    break
                } else {
                    ### Obtain the RRBLUP coefficients
                    coeffsRRBLUP <- mixed.new$u
                    ### Get fitted values of RRBLUP
                    fittedRRBLUP <- c(
                        as.matrix(X.test[, xVariables, with = F]) %*%
                            coeffsRRBLUP
                    ) +
                        mixed.new$beta
                    ### Calculate and store RRBLUP accuracy
                    accuracyPerScenario[k, 1] <- signif(
                        cor(Y.test, fittedRRBLUP),
                        digits = 3
                    )
                }

                ### Store accuracy for this run
                accuracy[k, j + 1] <- accuracyPerScenario[k, 1]
                pb_prop$tick()
            }

            j <- j + 1
            pb_runs$tick(tokens = list(what = paste("Run", j)))
            overallAccuracy <- overallAccuracy + accuracyPerScenario
        }

        overallAccuracy <- overallAccuracy / j
        objectsToReturn[[yVar]] <- list(
            "accuracy" = overallAccuracy,
            "stats" = accuracy # Match psBLUP_raw.R output structure
        )
        pb_yvar$tick()
    }
    cat("\nrrBLUP processing completed!\n")

    # to extract the accuracy of rrBLUP you can use the following code:
    # accuracy_rr <- objectsToReturn[[1]]$accuracy[1,1]
    # accuracy_stats <- objectsToReturn[[1]]$stats
    return(objectsToReturn)
}
