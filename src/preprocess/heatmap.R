setwd("./")
source("data_preprocess.R")

numeric_vars <- c(con_vars, response)
df <- bccp[numeric_vars]

library(caret)
dummy_data <- dummyVars(~., data = df)
bank_data_dummy <- data.frame(predict(dummy_data, newdata = df))

library(corrplot)
corr_matrix <- cor(bank_data_dummy)
corrplot(corr_matrix, tl.cex = 0.8,tl.col = "blue")