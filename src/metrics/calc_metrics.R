library(pROC)

# Calculate confusion matrix
get_confusion_matrix = function(actual_response, predicted_response, t) {
  pred_class <- ifelse(predicted_response > t, 1, 0)
  cm <- table(pred_class, actual_response)
  cm
}

# Calulate the AUC
getAUC = function(actual_response, predicted_response) {
  roc_obj <- roc(actual_response, predicted_response)
  model_auc = auc(roc_obj)
  model_auc
}

# plot ROC curve
plot_roc = function(actual_response, predicted_response) {
  roc_obj <- roc(actual_response, predicted_response)
  plot(roc_obj)
}

# This function is used to return the best tune threshold for classification.
get_opt_metrics = function(actual_response, 
                           predicted_response, 
                           target_metric) {
  acc = c()
  for(t in seq(0, 1, 0.01)) {
    cm <- get_confusion_matrix(actual_response, predicted_response, t)
    if (ncol(cm) == 2 && nrow(cm) == 2 && 
        sum(cm[2, ]) > 0 && sum(cm[, 2]) >0 &&
        sum(cm[1, ]) > 0 && sum(cm[, 1]) > 0) {
      # total accuracy
      cur_acc = (cm[1, 1] + cm[2, 2]) / length(actual_response)
      
      # Positive precision
      pprecision = cm[2, 2] / sum(cm[2, ])
      precall = cm[2, 2] / sum(cm[, 2])
      
      # Negative precision
      nprecision = cm[1, 1] / sum(cm[1, ])
      nrecall = cm[1, 1] / sum(cm[, 1])
      
      acc = rbind(acc, c(t, cur_acc, 
                         pprecision, precall, 
                         nprecision, nrecall))
    }
  }
  
  acc_df = data.frame(acc)
  colnames(acc_df) <- c("threshold", "total_accuracy", 
                        "positive_precision", "positive_recall", 
                        "negative_precision", "negative_recall")
  best_tune_row = acc_df[which.max(acc_df[, c(target_metric)]), ]
  
  # All metrics
  cm <- get_confusion_matrix(actual_response, 
                             predicted_response, 
                             best_tune_row$threshold)
  model_metrics = list(
    threshold = best_tune_row$threshold,
    total_accuracy = best_tune_row$total_accuracy,
    confusion_matrix = cm,
    positive_precision = best_tune_row$positive_precision,
    positive_recall = best_tune_row$positive_recall,
    negative_precision = best_tune_row$negative_precision,
    negative_recall = best_tune_row$negative_recall
  )
  model_metrics
}

get_metrics_by_threshold = 
  function(actual_response, predicted_response, t) {
  cm <- get_confusion_matrix(actual_response, predicted_response, t)
  
  # total accuracy
  cur_acc = (cm[1, 1] + cm[2, 2]) / length(actual_response)
    
  # Positive precision
  pprecision = cm[2, 2] / sum(cm[2, ])
  precall = cm[2, 2] / sum(cm[, 2])
    
  # Negative precision
  nprecision = cm[1, 1] / sum(cm[1, ])
  nrecall = cm[1, 1] / sum(cm[, 1])
    
  # All metrics
  model_metrics = list(
    threshold = t,
    total_accuracy = cur_acc,
    confusion_matrix = cm,
    positive_precision = pprecision,
    positive_recall = precall,
    negative_precision = nprecision,
    negative_recall = nrecall
  )
  model_metrics
}