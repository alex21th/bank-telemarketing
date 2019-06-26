library(class)
library(caret)
library(MASS)
library(MLmetrics)
library(klaR)
library(DMwR)
library(rpart)
library(rpart.plot)
library(rattle)
library(tree)
library(randomForest)
library(kernlab)



###
# verbose = TRUE
###

# Function that prints a beautiful interpretation of prediction accuracies
print_sol <- function(confusion.matrix.tr, confusion.matrix.te, f1.tr, f1.te, onlytest = FALSE) {
  if (onlytest == FALSE) {
    cat(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n')
    cat('Total training accuracy: ', sum(diag(confusion.matrix.tr))/sum(confusion.matrix.tr), '\n')
    cat('Confusion matrix:')
    print(confusion.matrix.tr)
    cat('Positive accuracy: ', confusion.matrix.tr[4]/colSums(confusion.matrix.tr)[2], ' , True negatives: ', confusion.matrix.tr[2], '\n')
    cat('F1 score: ', f1.tr, '\n')
  }
  # Print testing results
  cat(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n')
  cat('Total testing accuracy: ', sum(diag(confusion.matrix.te))/sum(confusion.matrix.te), '\n')
  cat('Confusion matrix:')
  print(confusion.matrix.te)
  cat('Positive accuracy: ', confusion.matrix.te[4]/colSums(confusion.matrix.te)[2], ' , True negatives: ', confusion.matrix.te[2], '\n')
  cat('F1 score: ', f1.te, '\n')
  cat(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n')
}



###
# KNN
###

# Notice we cannot execute the following comand:
#     proves <- kNN(y ~ ., training, testing, k = 1)
# It is required that all our data is numeric: so we need to convert our factors to numbers. We will use dummy variables.
# Notice that all of the functions, such as the ones for neural networks, lda, qda, rda, etc. already can handle categorical variables by themselves, so we will only use this following function for the kNN approach.
dummy <- function(dd) {
  dd.num <- dd
  # We binarize the categoral variables with only two classes
  dd.num$default <- ifelse(dd.num$default == 'yes', 1, 0)
  dd.num$housing <- ifelse(dd.num$housing == 'yes', 1, 0)
  dd.num$loan    <- ifelse(dd.num$loan    == 'yes', 1, 0)
  dd.num$contact <- ifelse(dd.num$contact == 'telephone', 1, 0)
  dd.num$y       <- ifelse(dd.num$y       == 'yes', 1, 0)
  # Binarize the rest; that is, the categoral variables with more than two classes
  dd.num <- data.frame(predict(dummyVars('~.', data = dd.num), newdata = dd.num))
  return(dd.num)
}
# Now we will be able to perform kNN predictions with the following modifications:
#     proves <- kNN(y ~ ., dummy(training), dummy(testing), k = 1)


# k is the number of nearest neighbours to be considered
function_knn <- function(training, testing,
                         k = 3, verbose = FALSE) {
  # Binarize categorical variables (with previous function)
  dummytr <- data.frame(dummy(training))
  dummyte <- data.frame(dummy(testing))
  # Fit the model
  fit <- kNN(y ~ ., train = dummytr, test = dummyte, k = k)
  # Check predictions
  confusion.matrix.te <- table(fit, testing$y)
  # Print solution
  if (verbose == TRUE) {
    print_sol(NULL, confusion.matrix.te, NULL, NULL, onlytest = TRUE)
  }
  
  return(list('confusion.matrix.te' = confusion.matrix.te))
}



###
# DISCRIMINANT ANALYSIS
###

# Function that takes a training+testing set and performs DISCRIMINANT ANALYSIS.
# Method = 'linear', 'quadratic', 'regularized'
# It returns a table with both training and test predictions + 
function_DA <- function(training, testing,
                        method = c('linear', 'quadratic', 'regularized'),
                        prior = NULL, verbose = FALSE) {
  options(warn = -1)
  # Fit the model
  if (is.null(prior)) {
    if (method == 'linear') {
      fit <- lda(y ~ ., data = training)
    } else if (method == 'quadratic') {
      fit <- qda(y ~ ., data = training)
    } else {
      fit <- rda(y ~ ., data = training)
    }
  } else {
    if (method == 'linear') {
      fit <- lda(y ~ ., data = training, prior = prior)
    } else if (method == 'quadratic') {
      fit <- qda(y ~ ., data = training, prior = prior)
    } else {
      fit <- rda(y ~ ., data = training, prior = prior)
    }
  }
  # Predict on training data
  fit.tr_pred     <- predict(fit, newdata = training)$class
  confusion.matrix.tr <- table(fit.tr_pred, training$y)
  # Predict on testing data
  fit.te_pred     <- predict(fit, newdata = testing)$class
  confusion.matrix.te <- table(fit.te_pred, testing$y)
  # Compute F1 score
  f1.tr <- F1_Score(fit.tr_pred, training$y, positive = 'yes')
  f1.te <- F1_Score(fit.te_pred, testing$y, positive = 'yes')
  # Print solution
  if (verbose == TRUE) {
    print_sol(confusion.matrix.tr, confusion.matrix.te, f1.tr, f1.te)
  }
  
  options(warn = 0)
  return(list('confusion.matrix.tr' = confusion.matrix.tr, 'f1.tr' = f1.tr, 'confusion.matrix.te' = confusion.matrix.te, 'f1.te' = f1.te))
}



###
# LOGISTIC REGRESSION
###

function_logreg <- function(training, testing,
                            posterior = .5, verbose = FALSE) {
  options(warn = -1)
  # Fit the model
  fit <- glm(y ~ ., data = training, family = 'binomial')
  # Predictions on training data
  fit.tr_probs <- predict(fit, newdata = training, type = 'response')
  fit.tr_pred  <- rep('no', length(fit.tr_probs))
  fit.tr_pred[fit.tr_probs >= posterior] <- 'yes'
  confusion.matrix.tr <- table(fit.tr_pred, training$y)
  # Predictions on testing data
  fit.te_probs <- predict(fit, newdata = testing, type = 'response')
  fit.te_pred  <- rep('no', length(fit.te_probs))
  fit.te_pred[fit.te_probs >= posterior] <- 'yes'
  confusion.matrix.te <- table(fit.te_pred, testing$y)
  # Compute F1 score
  f1.tr <- F1_Score(fit.tr_pred, training$y, positive = 'yes')
  f1.te <- F1_Score(fit.te_pred, testing$y, positive = 'yes')
  # Print solution
  if (verbose == TRUE) {
    print_sol(confusion.matrix.tr, confusion.matrix.te, f1.tr, f1.te)
  }
  # Return the accuracies for ploting
  tr.acc <- mean(fit.tr_pred == training$y)
  te.acc <- mean(fit.te_pred == testing$y)
  tr.yes.acc <- confusion.matrix.tr[4]/colSums(confusion.matrix.tr)[2]
  te.yes.acc <- confusion.matrix.te[4]/colSums(confusion.matrix.te)[2]
  tr.tn <- confusion.matrix.tr[2]
  te.tn <- confusion.matrix.te[2]
  
  options(warn = 0)
  return(c(tr.acc, tr.yes.acc, tr.tn, f1.tr, te.acc, te.yes.acc, te.tn, f1.te))
}



###
# NAIVE BAYES
###

function_naiveBayes <- function(training, testing,
                                prior = NULL, verbose = FALSE) {
  options(warn = -1)
  # Fit the model
  if (is.null(prior)) {
    fit <- NaiveBayes(y ~ ., data = training)
  } else {
    fit <- NaiveBayes(y ~ ., data = training, prior = prior)
  }
  # Predict on training data
  fit.tr_pred <- predict(fit, newdata = training)$class
  confusion.matrix.tr <- table(fit.tr_pred, training$y)
  # Predict on testing data
  fit.te_pred     <- predict(fit, newdata = testing)$class
  confusion.matrix.te <- table(fit.te_pred, testing$y)
  # Compute F1 score
  f1.tr <- F1_Score(fit.tr_pred, training$y, positive = 'yes')
  f1.te <- F1_Score(fit.te_pred, testing$y, positive = 'yes')
  # Print solution
  if (verbose == TRUE) {
    print_sol(confusion.matrix.tr, confusion.matrix.te, f1.tr, f1.te)
  }
  
  options(warn = 0)
  return(list('confusion.matrix.tr' = confusion.matrix.tr, 'f1.tr' = f1.tr, 'confusion.matrix.te' = confusion.matrix.te, 'f1.te' = f1.te))
}



###
# DECISSION TREES
###

function_CART <- function(training, testing,
                          parms, verbose = FALSE, plot = FALSE) {
  # First we need to fit the model
  fit <- rpart(y ~ ., data = training, parms = parms)
  # Predict on training data
  fit.tr_pred <- predict(fit, newdata = training, type = 'class')
  confusion.matrix.tr <- table(fit.tr_pred, training$y)
  # Predict on testing data
  fit.te_pred <- predict(fit, newdata = testing, type = 'class')
  confusion.matrix.te <- table(fit.te_pred, testing$y)
  # Compute F1 score
  f1.tr <- F1_Score(fit.tr_pred, training$y, positive = 'yes')
  f1.te <- F1_Score(fit.te_pred, testing$y, positive = 'yes')
  # Print solution
  if (verbose == TRUE) {
    print_sol(confusion.matrix.tr, confusion.matrix.te, f1.tr, f1.te)
  }
  # If plot == true
  if (plot == TRUE) {
    fancyRpartPlot(fit)
  }
  
  return(list('confusion.matrix.tr' = confusion.matrix.tr, 'f1.tr' = f1.tr, 'confusion.matrix.te' = confusion.matrix.te, 'f1.te' = f1.te))
}



###
# RANDOM FORESTS
###

function_randf <- function(training, testing,
                           classwt = NULL, ntree = 500,
                           mtry = NULL, nodesize = 1,
                           verbose = FALSE) {
  # Fit the model
  if (is.null(mtry)) {
    fit <- randomForest(y ~ ., data = training,
                        ntree = ntree, classwt = classwt, 
                        nodesize = nodesize)
  } else {
    fit <- randomForest(y ~ ., data = training,
                        ntree = ntree, mtry = mtry,
                        classwt = classwt, nodesize = nodesize)
  }
  # Predict on training data
  fit.tr_pred <- predict(fit, newdata = training)
  confusion.matrix.tr <- table(fit.tr_pred, training$y)
  # Predict on testing data
  fit.te_pred     <- predict(fit, newdata = testing)
  confusion.matrix.te <- table(fit.te_pred, testing$y)
  # Compute F1 score
  f1.tr <- F1_Score(fit.tr_pred, training$y, positive = 'yes')
  f1.te <- F1_Score(fit.te_pred, testing$y, positive = 'yes')
  # Print solution
  if (verbose == TRUE) {
    print_sol(confusion.matrix.tr, confusion.matrix.te, f1.tr, f1.te)
  }
  
  return(list('confusion.matrix.tr' = confusion.matrix.tr, 'f1.tr' = f1.tr, 'confusion.matrix.te' = confusion.matrix.te, 'f1.te' = f1.te))
}



###
# SUPPORT VECTOR MACHINES
###

# svm implementation using ksvm{kernlab}
# kernel must be one of (rbfdot, polydot, vanilladot, tanhdot, laplacedot, besseldot, anovadot, splinedot, stringdot)
function_svm <- function(training, testing,
                         kernel, kpar = 'automatic', C = 1,
                         verbose = FALSE) {
  # Fit the model
  fit <- ksvm(y ~ ., data = training, type = 'C-svc',
              kernel = kernel, kpar = kpar, C = C)
  # Predict on training data
  fit.tr_pred <- predict(fit, newdata = training)
  confusion.matrix.tr <- table(fit.tr_pred, training$y)
  # Predict on testing data
  fit.te_pred <- predict(fit, newdata = testing)
  confusion.matrix.te <- table(fit.te_pred, testing$y)
  # Compute F1 score
  f1.tr <- F1_Score(fit.tr_pred, training$y, positive = 'yes')
  f1.te <- F1_Score(fit.te_pred, testing$y, positive = 'yes')
  # Print solution
  if (verbose == TRUE) {
    print_sol(confusion.matrix.tr, confusion.matrix.te, f1.tr, f1.te)
  }
  
  return(list('confusion.matrix.tr' = confusion.matrix.tr, 'f1.tr' = f1.tr, 'confusion.matrix.te' = confusion.matrix.te, 'f1.te' = f1.te))
}


################################################################################
# NEURAL NETWORK (NN)
################################################################################

# We play with the diferent training sets to fit our NN.
function_neuralNet <- function(training, testing,
                               decay, verbose = FALSE) {
  # Fit the model
  fit <- nnet(y ~., data = training, size=H, maxit=200, MaxNWts = 1200, trace = F, decay=decay)
  
  # Predictions on training data
  fit.tr_pred <- as.factor(predict(fit, newdata = training, type = 'class'))
  confusion.matrix.tr <- table(fit.tr_pred, training$y)
  
  # Predictions on testing data
  fit.te_pred <- as.factor(predict(fit, newdata = testing, type = 'class'))
  confusion.matrix.te <- table(fit.te_pred, testing$y)
  
  # Compute F1 score
  f1.tr <- F1_Score(fit.tr_pred, training$y, positive = 'yes')
  f1.te <- F1_Score(fit.te_pred, testing$y, positive = 'yes')
  # Print solution
  if (verbose == TRUE) {
    print_sol(confusion.matrix.tr, confusion.matrix.te, f1.tr, f1.te)
  }
  
  # Return the accuracies for ploting
  tr.acc <- mean(fit.tr_pred == training$y)
  te.acc <- mean(fit.te_pred == testing$y)
  tr.yes.acc <- confusion.matrix.tr[4]/colSums(confusion.matrix.tr)[2]
  te.yes.acc <- confusion.matrix.te[4]/colSums(confusion.matrix.te)[2]
  tr.tn <- confusion.matrix.tr[2]
  te.tn <- confusion.matrix.te[2]
  
  options(warn = 0)
  return(c(tr.acc, tr.yes.acc, tr.tn, f1.tr, te.acc, te.yes.acc, te.tn, f1.te))
}





