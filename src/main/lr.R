### Init env
setwd("./")
source("../preprocess/data_preprocess.R")
source("../metrics/calc_metrics.R")

### Fit the logistic regression model with training set.
## Not yetfeature selection in here.
full_train_bccp <- data.frame(train_x, train_y)
colnames(full_train_bccp)[length(full_train_bccp)] <- "y"
full_model <- glm(y ~ ., 
                  data = full_train_bccp, 
                  family = "binomial")
summary(full_model)

# Prediction
full_test_bccp <- data.frame(test_x, test_y)
colnames(full_test_bccp)[length(full_test_bccp)] <- "y"
full_pred <- predict(full_model, 
                     newdata = full_test_bccp, 
                     type = "response")

# Evaluate
target_metric = "total_accuracy"
opt_full_model_metrics = get_opt_metrics(
  as.integer(test_y) - 1, full_pred, target_metric)

auc = getAUC(as.integer(test_y) - 1, full_pred)
opt_full_model_metrics <- c(AUC = auc, 
                            opt_full_model_metrics)
opt_full_model_metrics

# We can also draw the ROC curve
plot_roc(as.integer(test_y) - 1, full_pred)


### Cross validation on whole set to check the robustness.
library(caret)
ntimes <- 10
kfold <- 5

# Use whole dataset
for_cv_bccp <- data.frame(x, y)

# set up cross-validation
train_control <- trainControl(method = "cv", 
                              number = kfold)

accs <- matrix(nrow = ntimes, ncol = kfold)
# train logistic regression model
for (i in seq(1: ntimes)) {
  model <- train(y ~ ., data = for_cv_bccp, 
                 method = "glm", 
                 family = "binomial", 
                 trControl = train_control)
  # view model results
  accs[i, ] <- model$resample$Accuracy
}

# Plot the matrix
plot(col(accs)[1,], accs[1,], type = "l", 
     xlab = "Test time of each cross validation", 
     ylab = "Accuracy on threshold 0.5", 
     ylim = c(0.9, 0.92), 
     col = rgb(1, 0, 0, 0.55))

# Add lines for the remaining rows
for (i in 2:nrow(accs)) {
  lines(col(accs)[i,], accs[i,], 
        col = rgb(1, 0, 0, 0.5 + 0.05 * i))
}

# Calculate the accuracy of the model in Scenario I, 
# set threshold as 0.5.
aboveAcc = get_metrics_by_threshold(
  as.integer(test_y) - 1, full_pred, 0.5)$total_accuracy
abline(h = aboveAcc, col = "black")