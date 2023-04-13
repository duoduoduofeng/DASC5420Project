library(ggplot2)

setwd("./")
source("data_preprocess.R")

all_cat_vars <- c(cat_vars, unused_cat_vars)
df <- bccp_ori[, all_cat_vars]
new_dfs <- list()
for(pred in all_cat_vars) {
  new_dfs[[pred]] <- data.frame(variable = rep(pred), 
                                value = df[[pred]])
}

cat_df <- NULL
for (pred in all_cat_vars) {
  cat_df <- rbind(cat_df, new_dfs[[pred]])
}

ggplot(data = cat_df) +
  geom_bar(mapping = aes(x = variable, 
                         fill = value), 
           position = "fill") + 
  labs(y="Proportion") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


ggplot(data = cat_df) +
  geom_bar(mapping = aes(x = variable, 
                         fill = value), 
           position = "fill") + 
  scale_fill_discrete(labels = 
                        unique(paste0(cat_df$variable, 
                                      "_", 
                                      cat_df$value))) + 
  theme(axis.text.x = 
          element_text(angle = 45, hjust = 1))














library(ggplot2)
library(tidyr)
library(dplyr)

setwd("./")
source("data_preprocess.R")

all_cat_vars <- c(cat_vars, unused_cat_vars)
df <- bccp_ori[, all_cat_vars]

# reshape data to long format
cat_df_long <- gather(df, key = variable, value = value)

cat_df_ratio <- df %>%
  gather(key = variable, value = value) %>%
  group_by(variable, value) %>%
  summarise(n = n()) %>%
  ungroup() %>%
  mutate(ratio = n/sum(n))

ggplot(cat_df_ratio, aes(x = variable, y = ratio, fill = value)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_y_continuous(labels = scales::percent_format()) +
  facet_wrap(~ variable, ncol = 5)

# create ratio bar plots for each categorical variable
ggplot(cat_df_long, aes(x = value, fill = variable)) +
  geom_bar(position = "stack") +
  facet_wrap(~ variable, ncol = 5) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
