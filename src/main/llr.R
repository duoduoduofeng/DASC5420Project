### Init env
setwd("./")
source("../preprocess/data_preprocess.R")
source("../metrics/calc_metrics.R")


### Apply lasso for feature selection
library(glmnet)
library(pROC)

# Find optimal lambda value using 10-fold cross-validation
set.seed(theseed)
cv_lasso <- cv.glmnet(train_x, 
                      train_y, 
                      family = "binomial", 
                      alpha = 1, 
                      type.measure = "class", 
                      nfolds = 10)

# Fit the Ridge and Lasso models using optimal lambda values
opt_lambda_lasso <- cv_lasso$lambda.min
lasso_model <- glmnet(train_x, 
                      train_y, 
                      family = "binomial", 
                      alpha = 1, 
                      lambda = opt_lambda_lasso)

# best tune lasso model information
best_lasso = list(
  opt_lambda_lasso = opt_lambda_lasso,
  opt_lasso_coefficients = coef(lasso_model)
)
best_lasso

# Prediction
lasso_pred <- predict(lasso_model, 
                      s = opt_lambda_lasso, 
                      newx = test_x)

# Evaluate
target_metric = "total_accuracy"
opt_lasso_model_metrics = get_opt_metrics(
  as.integer(test_y) - 1, lasso_pred, target_metric)

auc = getAUC(as.integer(test_y) - 1, lasso_pred)
opt_lasso_model_metrics <- c(AUC = auc, 
                             opt_lasso_model_metrics)

opt_lasso_model_metrics

# We can also draw the ROC curve
plot_roc(as.integer(test_y) - 1, lasso_pred)