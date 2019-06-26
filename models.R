# Set current directory to file location
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()

# Load data
load(paste0(getwd(), '/dd_complete.RData'))
# Data dimensionality; hence p is number of predictors (without response value)
N <- nrow(dd.complete); p <- ncol(dd.complete) -1
str(dd.complete)



# Set seed to reproduce obtained results
set.seed(123456)

# Current dataset
str(dd.complete)
any(is.na(dd.complete)) # No NA's after random forest imputation


###
# RESERVING FINAL TEST DATA
###

# We need to reserve some of our data in order to do our final test
library(caret)
# We select random 20% of our data for the final predictions
final_test <- createDataPartition(dd.complete$y, p = .2, list = FALSE)
train      <- dd.complete[-final_test,]
final_test <- dd.complete[final_test,]

###
# VALIDATION SET APPROACH
###

# This approach consists in splitting the data into two groups, a training set and a testing set.

training <- createDataPartition(train$y, p = .8, list = FALSE)
testing  <- train[-training,]
training <- train[training,]

# Check partition
summary(training$y); summary(testing$y)
# We have randomly split the data

# dd_complete = train U final_test, train: (training U testing)

###
# HANDLE IMBALANCED DATA
###

summary(dd.complete$y); summary(dd.complete$y) / N
# Notice that only an 11% of our data falls into category 'yes'

# Imbalanced classification is a supervised learning problem where one class outnumbers other class by a large proportion.
# The term imbalanced refer to the disparity encountered in the dependent (response) variable. Therefore, an imbalanced classification problem is one in which the dependent variable has imbalanced proportion of classes.
# Below are the reasons which leads to reduction in accuracy of ML algorithms on imbalanced data sets:
# - ML algorithms struggle with accuracy because of the unequal distribution in dependent variable.
# - This causes the performance of existing classifiers to get biased towards majority class.
# - The algorithms are accuracy driven i.e. they aim to minimize the overall error to which the minority class contributes very little.
# - ML algorithms assume that the data set has balanced class distributions.
# - They also assume that errors obtained from different classes have same cost (explained below in detail).

# So we have to adress this problem; in the code below we are going to try different aproaches to try to balance our data.

library(ROSE)
library(DMwR)

###
summary(training$default)
# Notice we only have one variable with 'default' = yes, so we have to ensure that every new partition has at least this observation, because otherwise we will get weird errors.
defaultyes <- training[which(training$default == 'yes'),]
###

# Oversampling
# This method is used to oversample minority class until it reaches the same number of observations as the majority class.
size <- nrow(training) + (length(which(training$y == 'no')) - length(which(training$y == 'yes')))
dd_balanced_over <- ovun.sample(y ~ ., data = training, method = 'over', N = size)$data
table(dd_balanced_over$y)
# We balanced our data
summary(dd_balanced_over$default)

# Undersampling
# We undersample without replacement the majority class
size <- length(which(training$y == 'yes'))*2
dd_balanced_under <- ovun.sample(y ~ ., data = training, method = 'under', N = size)$data
table(dd_balanced_under$y)
# We balanced our data
summary(dd_balanced_under$default) # We have no observations of class 'yes' for variable 'default': that would lead us to errors in logistic regression and other algorithms
# So we input an observation to the dataset:
dd_balanced_under <- rbind(dd_balanced_under, defaultyes)
summary(dd_balanced_under$default)

# Both sampling
# This method is a combination of both oversampling and undersampling methods. The majority class is undersampled without replacement and the minority class is oversampled with replacement.
size <- nrow(training)
dd_balanced_both <- ovun.sample(y ~ ., data = training, method = 'both', p = .5, N = size)$data
table(dd_balanced_both$y)
# We balanced our data
summary(dd_balanced_both$default) # We have no observations of class 'yes' for variable 'default': that would lead us to errors in logistic regression and other algorithms
# So we input an observation to the dataset:
dd_balanced_both <- rbind(dd_balanced_both, defaultyes)
summary(dd_balanced_both$default)

# ROSE sampling
# Rose sampling method generates data synthetically and provides a better estimate of original data.
dd_balanced_rose <- ROSE(y ~ ., data = training)$data
table(dd_balanced_rose$y)
# We balanced our data
summary(dd_balanced_rose$default) # We have no observations of class 'yes' for variable 'default': that would lead us to errors in logistic regression and other algorithms
# So we input an observation to the dataset:
dd_balanced_rose <- rbind(dd_balanced_rose, defaultyes)
summary(dd_balanced_rose$default)

# SMOTE
# Synthetic Minority Over-Sampling Technique is used to avoid overfitting when adding exact replicas of minority instances to the main dataset.
# (perc.over and perc.under) tuned to achieve a balanced dataset
dd_balanced_smote <- SMOTE(y ~ ., data = training, perc.over = 400, perc.under = 120)
table(dd_balanced_smote$y)
# We balanced our data
summary(dd_balanced_smote$default) # We have no observations of class 'yes' for variable 'default': that would lead us to errors in logistic regression and other algorithms
# So we input an observation to the dataset:
dd_balanced_smote <- rbind(dd_balanced_smote, defaultyes)
summary(dd_balanced_smote$default)


# Save data (security reasons)
save.image(paste0(getwd(), '/balanced.RData'))
load(paste0(getwd(), '/balanced.RData'))



###
# Functions
# We load all the functions designed by us using the R packages to implement a clear representation of training results
source('_functions_.R')
###


###
# KNN
###

# Standard dataset
knn1 <- function_knn(training, testing, k = 3, verbose = TRUE)
# Oversampling
knn2 <- function_knn(dd_balanced_over, testing, k = 3, verbose = TRUE)
# Undersampling
knn3 <- function_knn(dd_balanced_under, testing, k = 3, verbose = TRUE)
# Under-over sampling
knn4 <- function_knn(dd_balanced_both, testing, k = 10, verbose = TRUE)
# ROSE
knn5 <- function_knn(dd_balanced_rose, testing, k = 3, verbose = TRUE)
# SMOTE
knn6 <- function_knn(dd_balanced_rose, testing, k = 3, verbose = TRUE)



###
# LDA
###

# Standard dataset
lda1 <- function_DA(training, testing, verbose = TRUE)
# Make the prior probabilities go even
lda2 <- function_DA(training, testing, prior = c(.5,.5), verbose = TRUE)
# Try oversampling
lda3 <- function_DA(dd_balanced_over, testing, verbose = TRUE)
# Try undersampling
lda4 <- function_DA(dd_balanced_under, testing, verbose = TRUE)
# Try under-oversampling
lda5 <- function_DA(dd_balanced_both, testing, verbose = TRUE)
# ROSE
lda6 <- function_DA(dd_balanced_rose, testing, verbose = TRUE)
# SMOTE
lda7 <- function_DA(dd_balanced_smote, testing, verbose = TRUE)



###
# QDA
###

# Rank defficiency in all datasets
#     qda1 <- function_DA(training, testing, method = 'quadratic', verbose = TRUE)


###
# RDA
###

# Builds a classification rule using regularized group covariance matrices that are supposed to be more robust against multicollinearity in the data.

# Standard dataset
rda1 <- function_DA(training, testing, method = 'regularized', verbose = TRUE)
# Priors
rda2 <- function_DA(training, testing, method = 'regularized', prior = c(.5,.5), verbose = TRUE)
# Oversampling
rda3 <- function_DA(dd_balanced_over, testing, method = 'regularized', verbose = TRUE)
# Undersampling
rda4 <- function_DA(dd_balanced_under, testing, method = 'regularized', verbose = TRUE)
# Both
rda5 <- function_DA(dd_balanced_both, testing, method = 'regularized', verbose = TRUE)
# ROSE
rda6 <- function_DA(dd_balanced_rose, testing, method = 'regularized', verbose = TRUE)
# SMOTE
rda7 <- function_DA(dd_balanced_smote, testing, method = 'regularized', verbose = TRUE)

# Notice we get the exact same results modifying the priors and with all the other balanced datasets.


###
# NAIVE BAYES CLASSIFIER
###

# Standard dataset
naive1 <- function_naiveBayes(training, testing, verbose = TRUE)
# Priors
naive2 <- function_naiveBayes(training, testing, prior = c(.5,.5), verbose = TRUE)
# Oversampling
naive3 <- function_naiveBayes(dd_balanced_over, testing, verbose = TRUE)
# Undersampling
naive4 <- function_naiveBayes(dd_balanced_under, testing, verbose = TRUE)
# Under-over sampling
naive5 <- function_naiveBayes(dd_balanced_both, testing, verbose = TRUE)
# ROSE
naive6 <- function_naiveBayes(dd_balanced_rose, testing, verbose = TRUE)
# SMOTE
naive7 <- function_naiveBayes(dd_balanced_smote, testing, verbose = TRUE)



###
# LOGISTIC REGRESSION
###

# We play with posterior probabilities with the diferent training sets to fit our logistic regression
posterior.play <- function(training, testing) {
  # All the posterior probabilities we are going to try
  posteriors <- seq(.5, .05, by = -.05)
  values <- c('tr.acc', 'tr.acc.yes', 'tr.tn', 'f1.tr', 'te.acc', 'te.acc.yes', 'te.tn', 'f1.te')
  # Matrix that stores all the results
  comparision <- matrix(ncol = 8, nrow = 10, dimnames = list(posteriors, values))
  for (i in 1:10) {
    comparision[i,] <- function_logreg(training, testing, posterior = posteriors[i])
  }
  # PLOTS
  par(mfcol = c(1,2))
  # Training set plots
  plot(x = posteriors, y = comparision[,1], ylim = c(0,1), xlim = c(0,.5), type = 'b', pch = 19, lty = 2, col = 'red', main = 'Training set', xlab = 'Posterior probabilities', ylab = '')
  lines(x = posteriors, y = comparision[,2], type = 'b', pch = 19, lty = 2, col = 'green')
  legend('bottomleft', legend = c('Total accuracy', 'Yes accuracy'), col = c('red', 'green'), lty = 2)
  
  # Testing set plots
  plot(x = posteriors, y = comparision[,5], ylim = c(0,1), xlim = c(0,.5), type = 'b', pch = 19, lty = 2, col = 'red', main = 'Testing set', xlab = 'Posterior probabilites', ylab = '')
  lines(x = posteriors, y = comparision[,6], type = 'b', pch = 19, lty = 2, col = 'green')
  legend('bottomleft', legend = c('Total accuracy', 'Yes accuracy'), col = c('red', 'green'), lty = 2)
  
  par(mfrow = c(1,1))
  return(comparision)
}

# Regular data set
logreg1 <- posterior.play(training, testing)
# Check test accuracies
logreg1[,c(5,6,7)]
# Check true-negatives and f1 scores
logreg1[,c(3,4,7,8)]

# Oversampling
logreg2 <- posterior.play(dd_balanced_over, testing)
# Check test accuracies
logreg2[,c(5,6,7)]
# Check true-negatives and f1 scores
logreg2[,c(3,4,7,8)]

# Undersampling
logreg3 <- posterior.play(dd_balanced_under, testing)
# Check test accuracies
logreg3[,c(5,6,7)]
# Check true-negatives and f1 scores
logreg3[,c(3,4,7,8)]

# Both
logreg4 <- posterior.play(dd_balanced_both, testing)
# Check test accuracies
logreg4[,c(5,6,7)]
# Check true-negatives and f1 scores
logreg4[,c(3,4,7,8)]

# ROSE
logreg5 <- posterior.play(dd_balanced_rose, testing)
# Check test accuracies
logreg5[,c(5,6,7)]
# Check true-negatives and f1 scores
logreg5[,c(3,4,7,8)]

# SMOTE
logreg6 <- posterior.play(dd_balanced_smote, testing)
# Check test accuracies
logreg6[,c(5,6,7)]
# Check true-negatives and f1 scores
logreg6[,c(3,4,7,8)]



###
# DECISSION TREES
###

# A Decision Tree is a supervised learning predictive model that uses a set of binary rules to calculate a target value.It is used for either classification (categorical target variable) or regression (continuous target variable). 

# Ara crearem una funci? de p?rdua per quan el nostre model classifica malament una observaci?.
# Loss matrix --> [TP  FP]
#                 [FN  TN]
loss  <- matrix(c(0,1,3,0), byrow = TRUE, ncol = 2)
loss2 <- matrix(c(0,1,2,0), byrow = TRUE, ncol = 2)

cart1.1 <- function_CART(training, testing, verbose = TRUE, plot = TRUE)
# Modify the priors
cart1.2 <- function_CART(training, testing, parms = list(prior = c(.5,.5)), verbose = TRUE, plot = TRUE)
# Try with one loss function
cart1.3 <- function_CART(training, testing, parms = list(loss = loss), verbose = TRUE, plot = TRUE)
# Combine both
cart1.4 <- function_CART(training, testing, parms = list(prior = c(.5,.5), loss = loss2), verbose = TRUE, plot = TRUE)
# All the other datasets have balanced classes, so that this is equivalent to the parameter prior = c(.5,.5) used beforehand
cart2  <- function_CART(dd_balanced_over, testing, verbose = TRUE)
cart3  <- function_CART(dd_balanced_under, testing, verbose = TRUE)
cart4  <- function_CART(dd_balanced_both, testing, verbose = TRUE)
cart5  <- function_CART(dd_balanced_rose, testing, verbose = TRUE)
cart6  <- function_CART(dd_balanced_smote, testing, verbose = TRUE)
# We can try to combine one of these methods with the loss functions (for example, the SMOTE with the loss)
cart3.1 <- function_CART(dd_balanced_smote, testing, parms = list(loss = loss), verbose = TRUE, plot = TRUE)

prova.cart <- function_CART(dd_balanced_under[,c(18,20)], testing[,c(18,20)], verbose = T, plot = T)
prova.cart <- function_CART(dd_balanced_under[,c(18,19,20)], testing[,c(18,19,20)], verbose = T, plot = T)

###
# RANDOM FORESTS
###

# Standard dataset
randf1 <- function_randf(training, testing, verbose = TRUE)
# Modify priors
randf1 <- function_randf(training, testing, classwt = c(.5,.5), verbose = TRUE)
# Span more trees
randf1 <- function_randf(training, testing, classwt = c(.5,.5), ntree = 1000, verbose = TRUE) #slighty increase global accuracy but lower 'yes' accuracy
# Different mtry
randf1 <- function_randf(training, testing, classwt = c(.5,.5), mtry = 15, verbose = TRUE) #lower 'yes' accuracy: higher total accuracy
# Different nodesize
randf1 <- function_randf(training, testing, classwt = c(.5,.5), nodesize = 60, verbose = TRUE)

# Oversampling
randf2 <- function_randf(dd_balanced_over, testing, verbose = TRUE)
# Undersampling
randf3 <- function_randf(dd_balanced_under, testing, verbose = TRUE)
# Under-over sampling
randf4 <- function_randf(dd_balanced_both, testing, verbose = TRUE)
# ROSE
randf5 <- function_randf(dd_balanced_rose, testing, verbose = TRUE)
# SMOTE
randf6 <- function_randf(dd_balanced_smote, testing, verbose = TRUE)

# Function randomForest{randomForest} implements a function varImp{} that let's you compute the importance of every variable, in order to perform feature selection:
aux <- randomForest(y ~ ., data = training)
varImpPlot(aux)
aux <- varImp(aux)
# The following plot suggests that we may try to fit our models without using all the variables available in the dataset. For instance, we can try our models with the ones with a value higher than 200 for example:
var7 <- c(which(aux$Overall >= 200), 20)
var4 <- c(which(aux$Overall >= 300), 20)
 
# After performing several experiments with the 'feature selected datasets', we conclude that the following are the ones that presented better results:
randf7   <- function_randf(dd_balanced_under[,var7], testing[,var7], nodesize = 60, verbose = TRUE)
randf7.1 <- function_randf(dd_balanced_under[,var4], testing[,var4], nodesize = 60, verbose = TRUE)
randf7.2 <- function_randf(dd_balanced_under[,c(18,19,20)], testing[,c(18,19,20)], nodesize = 80, verbose = TRUE)



###
# SVM
###

# Standard dataset
svm1   <- function_svm(training, testing, kernel = 'rbfdot', C = 1, verbose = TRUE)
# Under sampling
svm2   <- function_svm(dd_balanced_under, testing, kernel = 'rbfdot', C = 1, verbose = TRUE)
svm2.1 <- function_svm(dd_balanced_under, testing, kernel = 'laplacedot', C = 0.5, verbose = TRUE)
svm2.2 <- function_svm(dd_balanced_under[,c(18,19,20)], testing[,c(18,19,20)], kernel = 'laplacedot', C = 0.3, verbose = TRUE)
svm2.3 <- function_svm(dd_balanced_under[,c(18,20)], testing[,c(18,20)], kernel = 'laplacedot', C = 0.5, verbose = TRUE)
svm2.3 <- function_svm(dd_balanced_under[,c(18,20)], testing[,c(18,20)], kernel = 'tanhdot', kpar = list(scale = 0.5, offset = 2.5), C = 10, verbose = TRUE)

svm3   <- function_svm(dd_balanced_under[,c(18,19,20)], testing[,c(18,19,20)], kernel = 'laplacedot', C = 2, verbose = TRUE)



################################################################################
# NEURAL NETWORK (NN)
################################################################################

library(MASS)
library(nnet)  # For the NN itself.
library(caret) # For the CV of model parameters.
source('_functions_.R')

# Firstly, the 'nnet' function seems to do the mapping from a factor level to
# dummy variables internally and one doesn't need to be aware of it.

# We will use all five (dd_balanced_over, dd_balanced_under, dd_balanced_both,
# dd_balanced_rose, dd_balanced_smote) samples we've generated to try to balance
# our data.

# In order to find the best network architecture, we can explore two methods:
#     a) Explore different numbers of hidden units in one hidden layer,
#        with no regularization
#     b) Fix a number of H hidden units in one hidden layer and explore
#        different regularization values (RECOMMENDED and used here)

#     * Doing both a) AND b) is usually a waste of computing resources!

# --- Number of hidden neurons:
#       We set the number of hidden nodes using the heuristic H = round(M/2)
#       (M is the number of inputs) mentioned in the report.

# We don't really know how many inputs 'M' the NN creates for our data (due to
# the internal implementation of dummy variables). We get it:

useless.nnet <- nnet(y ~., data = training, size = 1, maxit = 200, decay = 0)
useless.nnet$n
M <- useless.nnet$n[1]
H <- round(M/2)

# --- Regularization values:
#       We set seven decay values (from 0.001 to 1) to test without using any CV
#       resampling method and the best one will be used to construct a final model.

# We play with the regularization values with different training sets to fit our NN.
decays.play <- function(training, testing) {
  # All the posterior probabilities we are going to try
  decays <- 10^seq(-3,0,by=0.5); n <- length(decays)
  values <- c('tr.acc', 'tr.acc.yes', 'tr.tn', 'f1.tr', 'te.acc', 'te.acc.yes', 'te.tn', 'f1.te')
  # Matrix that stores all the results
  comparision <- matrix(ncol = 8, nrow = n, dimnames = list(decays, values))
  for (i in 1:n) {
    comparision[i,] <- function_neuralNet(training, testing, decay = decays[i])
  }
  # PLOTS
  opar <- par(mfrow = c(1,2))
  # Training set plots
  plot(x = decays, y = comparision[,1], ylim = c(0,1), xlim = c(0,1), type = 'b', pch = 19, lty = 2, col = 'blue', main = 'Training set', xlab = 'Regularization values', ylab = '')
  lines(x = decays, y = comparision[,2], type = 'b', pch = 19, lty = 2, col = 'darkorange2')
  legend('bottomleft', legend = c('Total accuracy', 'Yes accuracy'), col = c('blue', 'darkorange2'), lty = 2)
  # Testing set plots
  plot(x = decays, y = comparision[,5], ylim = c(0,1), xlim = c(0,1), type = 'b', pch = 19, lty = 2, col = 'blue', main = 'Testing set', xlab = 'Regularization values', ylab = '')
  lines(x = decays, y = comparision[,6], type = 'b', pch = 19, lty = 2, col = 'darkorange2')
  legend('bottomleft', legend = c('Total accuracy', 'Yes accuracy'), col = c('blue', 'darkorange2'), lty = 2)
  par(opar)
  
  return(comparision)
}



## WARNING: each one takes around 20 minutes! ↓
# (simply load the saved results and skip that part)
# --> #load(paste0(getwd(), '/nnets.RData'))

# Training
nnet.training <- decays.play(training, testing)

# Oversampling
nnet.balanced_over <- decays.play(dd_balanced_over, testing)

# Undersampling
nnet.balanced_under <- decays.play(dd_balanced_under, testing)

# Both sampling
nnet.balanced_both <- decays.play(dd_balanced_both, testing)

# ROSE sampling
nnet.balanced_rose <- decays.play(dd_balanced_rose, testing)

# SMOTE sampling
nnet.balanced_smote <- decays.play(dd_balanced_smote, testing)

# Save data of the NN's (security reasons)
#save.image(paste0(getwd(), '/nnets.RData'))

# Now we can see the saved results (check test accuracies):
nnet.training[,c(5,6,7)]

nnet.balanced_over[,c(5,6,7)]

nnet.balanced_under[,c(5,6,7)]

nnet.balanced_both[,c(5,6,7)]

nnet.balanced_rose[,c(5,6,7)]

nnet.balanced_smote[,c(5,6,7)]



###
# RETRAIN THE BEST MODELS
###

# Here we will retrain the best models obtained in the last section.
# In order to do that, we will need to use all of the data usded for the previous training and testing, contained in the dataframe 'train':

# With standard dataset
retrain <- train

# Undersampling
size <- length(which(retrain$y == 'yes'))*2
retrain_under <- ovun.sample(y ~ ., data = retrain, method = 'under', N = size)$data
table(retrain_under$y)
# Now we have a balanced dataset with more observations
summary(retrain_under$default) # We have no observations of class 'yes' for variable 'default': that would lead us to errors in logistic regression and other algorithms
# So we input an observation to the dataset:
retrain_under <- rbind(retrain_under, defaultyes)
summary(retrain_under$default)

# Under-oversampling
size <- nrow(retrain)
retrain_both <- ovun.sample(y ~ ., data = retrain, method = 'both', p = .5, N = size)$data
table(retrain_both$y)
summary(retrain_both$default) # OK!


# NOW WE REFIT ALL THE BEST MODELS:

logreg <- function_logreg(retrain_both, final_test, verbose = TRUE)
bayes  <- function_naiveBayes(retrain, final_test, verbose = TRUE)
cart   <- function_CART(retrain_under, final_test, verbose = TRUE)
randf  <- function_randf(retrain_under, final_test, verbose = TRUE)
randf2 <- function_randf(retrain_under[,c(18,19,20)], final_test[,c(18,19,20)], nodesize = 10, verbose = TRUE)
svm    <- function_svm(retrain_under, final_test, kernel = 'rbf', C = 1, verbose = TRUE)
svm2   <- function_svm(retrain_under[,c(18,19,20)], final_test[,c(18,19,20)], kernel = 'laplacedot', C = 1, verbose = TRUE)
final.nnet <- decays.play(retrain_under, final_test)


