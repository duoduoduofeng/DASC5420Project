library(ggplot2)
setwd("./")
source("data_preprocess.R")

# create a sample data frame with numerical columns
df <- bccp_ori

# create a list of ggplot objects with separate axes and colored boxplots
colors <- c("#1F77B4", "#FF7F0E", "#2CA02C", 
            "#D62728", "#9467BD", "#8C564B")
plots <- lapply(con_vars, function(col) {
  ggplot(df, aes(x = col, y = !!as.name(col))) +
    geom_boxplot(fill = colors[match(col, con_vars)], 
                 color = colors[match(col, con_vars)]) +
    scale_y_continuous(name = col) +
    scale_fill_manual(values = colors)
})

# combine the ggplot objects into one plot using patchwork
library(patchwork)
plots[[1]] + plots[[2]] + plots[[3]] + 
  plots[[4]] + plots[[5]] + plots[[6]] + 
  plot_layout(nrow = 1)