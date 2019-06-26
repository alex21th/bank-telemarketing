# Set current directory to file location
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()

# Load data
load(paste0(getwd(), '/dd_complete.RData'))
# Data dimensionality; hence p is number of predictors (without response value)
N <- nrow(dd.complete); p <- ncol(dd.complete) -1
str(dd.complete)


library(cluster)
library(dplyr)
library(ggplot2)
library(readr)
library(Rtsne)

# data = dd.complete

# Clustering on a reduced portion of the dataset
example <- createDataPartition(dd.complete$y, p = .2, list = FALSE)
results <- dd.complete[example,20]
example <- dd.complete[example,-20]


# Compute Gower distance
gower_dist <- daisy(example, metric = 'gower')
gower_mat  <- as.matrix(gower_dist)

sil_width <- c(NA)
for (i in 2:10) {
  pam_fit <- pam(gower_dist, diss = TRUE, k = i)
  sil_width[i] <- pam_fit$silinfo$avg.width
}
plot(1:10, sil_width, xlab = 'Number of clusters',
     ylab = 'Silhouette Width', col = 'red')
lines(1:10, sil_width, col = 'red')

k <- 4
pam_fit <- pam(gower_dist, diss = TRUE, k)
pam_results <- example %>%
  mutate(cluster = pam_fit$clustering) %>%
  group_by(cluster) %>%
  do(the_summary = summary(.))
pam_results$the_summary

# Visualization in a lower dimensional space
tsne_obj  <- Rtsne(gower_dist, is_distance = TRUE)

tsne_data <- tsne_obj$Y %>%
  data.frame() %>%
  setNames(c('X','Y')) %>%
  mutate(cluster = factor(pam_fit$clustering))
ggplot(aes(x = X, y = Y), data = tsne_data) + geom_point(aes(color = cluster))

# Try to interpret our results with 2, 4 and 8 clusters

results2 <- pam_results$the_summary
table2   <- table(pam_fit$clustering, results)
results4 <- pam_results$the_summary
table4   <- table(pam_fit$clustering, results)
results8 <- pam_results$the_summary
table8   <- table(pam_fit$clustering, results)
# Visualize the results
results2
table2
results4
table4
results8
table8
# Proportion of class 'yes' in every cluster
table2[,2] / rowSums(table2) * 100
table4[,2] / rowSums(table4) * 100
table8[,2] / rowSums(table8) * 100
# It looks like clusters 3,4 (for k = 4) and 7,8 (for k = 8) have a much higher proportion of 'yes' ratio than the other clusters.
# So we conclude that there might be some characteristics of those clusters that specifically make costumers fall to category 'yes'.

# Try to visualize difference between cluster variables in order to classify between 'yes' or 'no
# We are going to take 4 as the number of clusters
aux <- cbind(example, pam_fit$clustering)
colnames(aux)[20] <- 'Cluster'
# Check the 'euribor' variance between clusters
boxplot(aux$euribor3m ~ aux$Cluster, main = 'Cluster comparision', xlab = 'Cluster', ylab = 'Euribor')
# Check the 'nr.employed' variance between clusters
boxplot(aux$nr.employed ~ aux$Cluster, main = 'Cluster comparision', xlab = 'Cluster', ylab = 'nr.employed')
# age
boxplot(aux$age ~ aux$Cluster)
