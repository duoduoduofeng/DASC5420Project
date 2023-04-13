library(ggplot2)

# Create a dataframe from the table
df <- data.frame(
  Situation = c("I", "II", "III", "IV", "V", "VI"),
  AUC = c(0.7563969, 0.7456696, 0.7575168, 0.7433761, 0.8623, 0.8465),
  Accuracy = c(0.811, 0.6928839, 0.8105, 0.690387, 0.8555, 0.779026),
  Positive_precision = c(0.6129032, 0.6477273, 0.6268657, 0.66, 0.510896, 0.763298),
  Positive_recall = c(0.2300242, 0.7579787, 0.2033898, 0.7021277, 0.708054, 0.765333),
  Negative_precision = c(0.8276423, 0.7479224, 0.823687, 0.7206983, 0.94518, 0.792941),
  Negative_recall = c(0.9621928, 0.6352941, 0.968494, 0.68, 0.881316, 0.79108)
)

# Melt the dataframe into long format
library(reshape2)
df_long <- melt(df, id.vars = "Situation")

# Order the situations by the smallest value of the selected metric
library(dplyr)
df_long <- df_long %>%
  arrange(value)

# Create the plot
ggplot(df_long, aes(x = variable, y = value, fill = Situation)) +
  geom_bar(stat = "identity", position = "identity", alpha = 0.8) +
  scale_fill_viridis_d() +
  labs(x = "Metric", y = "Value", 
       title = "Metrics under Different Situations") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"), 
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))













library(ggplot2)
library(tidyr)

# Create a data frame from the table
df <- data.frame(
  Situation = c("I", "II", "III", "IV", "V", "VI"),
  AUC = c(0.7563969, 0.7456696, 0.7575168, 0.7433761, 0.8623, 0.8465),
  Accuracy = c(0.811, 0.6928839, 0.8105, 0.690387, 0.855500, 0.779026),
  Positive_Precision = c(0.6129032, 0.6477273, 0.6268657, 0.66, 0.510896, 0.763298),
  Positive_Recall = c(0.2300242, 0.7579787, 0.2033898, 0.7021277, 0.708054, 0.765333),
  Negative_Precision = c(0.8276423, 0.7479224, 0.823687, 0.7206983, 0.945180, 0.792941),
  Negative_Recall = c(0.9621928, 0.6352941, 0.968494, 0.68, 0.881316, 0.791080)
)

# Convert the data from wide to long format for plotting
df_long <- df %>%
  pivot_longer(
    cols = c(AUC:Negative_Recall),
    names_to = "Metric",
    values_to = "Value"
  )

# Plot the data as a stacked bar plot with each metric as a separate bar
ggplot(df_long, aes(x = Metric, y = Value, fill = Situation)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  scale_fill_brewer(palette = "Set1") +
  labs(x = "Metric", y = "Value") +
  theme_minimal() +
  theme(legend.position = "bottom")








library(ggplot2)
library(reshape2)

# create a data frame with the given table
df <- data.frame(
  Situation = c("I", "II", "III", "IV", "V", "VI"),
  AUC = c(0.7563969, 0.7456696, 0.7575168, 0.7433761, 0.8623, 0.8465),
  Accuracy = c(0.811, 0.6928839, 0.8105, 0.690387, 0.855500, 0.779026),
  Positive_precision = c(0.6129032, 0.6477273, 0.6268657, 0.66, 0.510896, 0.763298),
  Positive_recall = c(0.2300242, 0.7579787, 0.2033898, 0.7021277, 0.708054, 0.765333),
  Negative_precision = c(0.8276423, 0.7479224, 0.823687, 0.7206983, 0.945180, 0.792941),
  Negative_recall = c(0.9621928, 0.6352941, 0.968494, 0.68, 0.881316, 0.791080)
)

metrics_long <- gather(df, metric, value, -Situation)

ggplot(metrics_long, aes(x = metric, y = value, 
                         fill = Situation)) + 
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Metrics under Different Situations", 
       x = "Metric", y = "Value") +
  theme(plot.title = element_text(hjust = 0.5, 
                                  size = 14, face = "bold"))