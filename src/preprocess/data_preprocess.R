### init env
setwd("./")
theseed = 542
need_balance = FALSE

### load data
bccp_ori = read.csv("../data/bank-additional-full.csv", 
                    sep = ";")

# shuffle the whole dataset
set.seed(theseed)
bccp_ori = bccp_ori[sample(nrow(bccp_ori)), ]

### Variables
response = "y"

cat_vars <- c("job", 
               "marital", 
               "education",
               "housing",
               "loan",
               "contact",
               "poutcome")
con_vars <- c("age",
              "duration",
              "campaign",
              "pdays",
              "previous",
              "emp.var.rate",
              "cons.price.idx",
              "cons.conf.idx")

unused_cat_vars = c("month",
                    "day_of_week", 
                    "default")
# To avoid singularization
unused_con_vars = c("euribor3m", 
                    "nr.employed") 


### Factor the response variable as {0, 1}
bccp_ori[[response]] = factor(bccp_ori[[response]], 
                              levels = c("yes", "no"), 
                              labels = c(1, 0))
head(bccp_ori)


# balancing
library(ROSE)
balanced_bccp = ovun.sample(y ~ ., 
                    data = bccp_ori, 
                    method = "under", 
                    seed = theseed)$data


### Choose original/balanced dataset to fit the model.
bccp = bccp_ori
if(need_balance) {
  bccp = balanced_bccp
}

### Convert factor column into dummy variables
factor_cols <- 
  model.matrix( ~ job + marital + education + 
                  default + housing + loan + 
                  contact + poutcome, 
                bccp)
bccp = data.frame(factor_cols[, 2:ncol(factor_cols)], 
                  bccp[, c(con_vars, response)])

### Data scaling
for (pred in con_vars) {
  bccp[[pred]] = scale(bccp[[pred]])
}

x <- as.matrix(bccp[, 2: (ncol(bccp)-1)])
y <- factor(bccp[, ncol(bccp)], 
            levels = c("0", "1"), 
            labels = c(0, 1))

### Data spliting
set.seed(theseed)
test_idx <- sample(nrow(bccp), nrow(bccp) * 0.2)
train_x <- x[-test_idx, ]
train_y <- y[-test_idx]
test_x <- x[test_idx, ]
test_y <- y[test_idx]

# write data into csv for python nn usage.
if (need_balance) {
  write.csv(balanced_bccp[-test_idx, ], 
            "../data/balanced_bcf_train.csv", 
            row.names = FALSE)
  write.csv(balanced_bccp[test_idx, ], 
            "../data/balanced_bcf_test.csv", 
            row.names = FALSE)
} else {
  write.csv(bccp_ori[-test_idx, ], 
            "../data/bcf_train.csv", 
            row.names = FALSE)
  write.csv(bccp_ori[test_idx, ], 
            "../data/bcf_test.csv", 
            row.names = FALSE)
}