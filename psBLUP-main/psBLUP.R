### psBLUB ----
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
  runs = 5,
  compareToRRBLUP = T
) {
  ##############################################################################
  # Arguments:
  # Y: data.frame of the response variables
  # X: data.frame of the predictor variables
  # proximityMatrix: A matrix with the proximity between predictor variables.
  #                  If its NULL, the observed correlation between the X
  #                  variables will be used
  # logTransformX: TRUE/FALSE on whether a log-transformation needs to be
  #                implemented on the response variables
  # percentTrainData: A vector with the percentages to be used for
  #                   training the model
  # l2s: A vector of penalties for inducing shrinkage on the differences between
  #      predictor variables
  # yVariables: a vector of names for the response variables to be used
  # xVariables: a vector of names for the predictor variables to be used
  # groupingVariables: a vector of names for variable included in the predictor
  #                    variables that we need to remove their effect, e.g.
  #                    treatment effects
  # runs: number of runs for calculating the final accuracy
  # compareToRRBLUP: TRUE/FALSE on whether the results need to be compared to
  #                  RRBLUP
  ##############################################################################

  ### Load relevant packages | were loaded before
  packages <- c("data.table", "plyr", "rrBLUP")
  lapply(packages, require, character.only = TRUE)

  ### Do the necessary checks for arguments
  if (is.null(X)) {
    stop("No predictor variables have been provided")
  }
  if (is.null(Y)) {
    stop("No dependent variables have been provided")
  }
  if (is.null(X)) {
    proximityMatrix <- cor(X)
  }
  if (!logTransformX %in% c(T, F)) {
    stop("Acceptable values for 'logTransformX': 'TRUE' or 'FALSE'")
  }
  if (!compareToRRBLUP %in% c(T, F)) {
    stop("Acceptable values for 'compareToRRBLUP': 'TRUE' or 'FALSE'")
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
  ### Transform data to data.table format
  setDT(X, keep.rownames = TRUE)
  setDT(Y, keep.rownames = TRUE)

  ### Keep only relevant variables
  Y <- Y[, c("rn", yVariables, groupingVariables), with = F]
  ### Keep only relevant variables
  X <- X[, c("rn", xVariables), with = F]

  ### Rename percentTrainData to proportion
  proportion = percentTrainData
  ### Apply log transformation to the data
  if (logTransformX) {
    Y[,
      c(yVariables) := lapply(.SD, function(x) {
        log(x)
      }),
      .SDcols = yVariables
    ]
  }

  ### Keep mean level of dependent variables per grouping variable
  ### Also remove grouping effect by regressing them out
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
      c(groupingVariables) := lapply(.SD, function(c) {
        NULL
      }),
      .SDcols = groupingVariables
    ]
    Y[,
      c(groupingVariables) := lapply(.SD, function(c) {
        NULL
      }),
      .SDcols = groupingVariables
    ]
  }

  ### Add strength (row-sum) as diagonal (needed for properly defining the Laplacian matrix)
  diag(proximityMatrix) <- apply(proximityMatrix, 1, sum)
  ### Scale the X variables
  X[,
    c(xVariables) := lapply(.SD, function(x) {
      scale(x)
    }),
    .SDcols = xVariables
  ]

  ### Define N as the number of samples
  N <- nrow(X)

  ### Run psBLUP for every metabolite
  for (yVar in yVariables) {
    ### define as "response" the scaled concentration  of the selected metabolite
    response <- c(scale(Y[, yVar, with = F]))
    ### define as "predictors" the SNP data
    predictors <- copy(X)
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

    ### Define the Normalized Laplacian matrix as G
    G <- proximityMatrix
    diagG <- diag(G)
    G <- (-G)
    diag(G) <- diagG
    ### Decompose the Normalize Laplacian matrix and keep the eigen-vectors and eigen-values
    eigenvalues <- eigen(G)$values
    eigenvalues[which(eigenvalues < 0)] <- 0
    eigenvectors <- eigen(G)$vectors
    ### Define as "newdata" the part that will be used for the augmented data solution
    newdata <- t(eigenvectors %*% diag(sqrt(eigenvalues)))
    colnames(newdata) <- rownames(newdata) <- xVariables
    ### Create a matrix to store all differences between RRBLUP and psBLUP per run
    accuracy <- matrix(NA, length(proportion), runs)
    colnames(accuracy) <- paste0("run: ", 1:runs)
    rownames(accuracy) <- paste0("train sample: ", proportion * 100, "%")

    ### Create a matrix containing the final accuracy results
    ### which will be the aggregated results over all runs
    overallAccuracy <- matrix(
      0,
      length(proportion),
      length(l2s) + compareToRRBLUP
    )
    colnames(overallAccuracy) <- c(
      if (compareToRRBLUP) {
        "RRBLUP"
      },
      paste0("l2 = ", signif(l2s, digits = 3))
    )
    rownames(overallAccuracy) <- paste0(
      "train = ",
      signif(proportion, digits = 2)
    )

    ### Run accuracy evaluation using while, as the samples might not contain genetic variation
    ### and produce error in fitting the models
    j <- 0
    while (j < runs) {
      ### Create a matrix for storing the accuracy per sampling scenario
      accuracyPerScenario <- matrix(
        NA,
        length(proportion),
        length(l2s) + compareToRRBLUP
      )
      colnames(accuracyPerScenario) <- c(
        if (compareToRRBLUP) {
          "RRBLUP"
        },
        paste0("l2 = ", signif(l2s, digits = 3))
      )
      rownames(accuracyPerScenario) <- paste0(
        "train = ",
        signif(proportion, digits = 2)
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

        if (compareToRRBLUP) {
          ### Fit RRBLUP
          mixed.new <- mixed.solve(
            Y.train,
            Z = X.train[, xVariables, with = F],
            method = "REML"
          )
          ### Obtain the RRBLUP coefficients
          coeffsRRBLUP <- mixed.new$u
          ### Get fitted values of RRBLUP
          fittedRRBLUP <- c(
            as.matrix(X.test[, xVariables, with = F]) %*% coeffsRRBLUP
          ) +
            mixed.new$beta
          ### Calculate and store RRBLUP accuracy on the correct cell
          accuracyPerScenario[k, 1] <- signif(
            cor(Y.test, fittedRRBLUP),
            digits = 3
          )
        }
        ### Run psBLUP for every l2 value
        for (i in 1:length(l2s)) {
          l2 = l2s[i]
          ### Calculate the factor to be premultiplied to the data for correctly defining the augmented data solution
          factor <- (1 + l2)^(-1 / 2)
          ### Create the augmented data for the predictors and response
          Xdata <- (factor *
            (rbind(X.train[, xVariables, with = F], sqrt(l2) * newdata)))
          Ydata <- (c(Y.train, rep(0, times = nrow(newdata))))
          Xdata <- data.frame(Xdata)
          ### Fit psBLUP
          mixed.new <- tryCatch(
            mixed.new <- mixed.solve(Ydata, Z = Xdata, method = "REML"),
            error = function(e) NA
          )
          ### Check if some model components have not been defined correctly
          if (sum(is.na(mixed.new)) == 1) {
            break
          } else {
            ### Obtain the psBLUP coefficients
            coeffsPSBLUP <- mixed.new$u
            ### Obtain the fitted values
            fittedPSBLUP <- c(
              as.matrix(X.test[, xVariables, with = F]) %*% coeffsPSBLUP
            ) +
              mixed.new$beta
            ### Calculate the psBLUP accuracy for a specific l2 and store it in the appropriate cell
            accuracyPerScenario[k, i + compareToRRBLUP] <- signif(
              cor(Y.test, fittedPSBLUP),
              digits = 3
            )
          }
        }
        ### For a specific scenario of training and testing data,
        ### the difference between RRBLUP and the max accuracy observed for a specific l2 is the accuracy gain
        accuracy[k, j + 1] <- max(accuracyPerScenario[
          k,
          (compareToRRBLUP + 1):ncol(accuracyPerScenario)
        ]) -
          accuracyPerScenario[k, 1] * compareToRRBLUP
      }
      j <- j + 1
      overallAccuracy <- overallAccuracy + accuracyPerScenario
    }
    overallAccuracy <- overallAccuracy / j
    objectsToReturn[[yVar]] <- list(
      "accuracy" = overallAccuracy,
      "gain" = accuracy
    )
  }
  # to extract the accuracy of psBLUP and RRBLUP you can use the following code:
  # accuracy_ps <- objectsToReturn[[1]]$accuracy[1,1]
  # accuracy_rr <- objectsToReturn[[1]]$accuracy[1,2]
  # accuracy_gain <- objectsToReturn[[1]]$gain
  return(objectsToReturn)
}
