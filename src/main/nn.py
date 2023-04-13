# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, \
     roc_curve, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

### Variables
response = "y"

cat_vars = ["job", 
            "marital", 
            "education",
            "housing",
            "loan",
            "contact",
            "poutcome"]
con_vars = ["age",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx"]

unused_cat_vars = ["month",
                   "day_of_week", 
                   "default"]
unused_con_vars = ["euribor3m", 
                    "nr.employed"]

# Init env
thebatchsize = 64
theepochtimes = 300
need_balance = True

### Load the dataset
train_df = pd.read_csv("../../data/bcf_train.csv")
test_df = pd.read_csv("../../data/bcf_test.csv")

if need_balance:
    thebatchsize = 64
    theepochtimes = 200

    train_df = pd.read_csv("../../data/balanced_bcf_train.csv")
    test_df = pd.read_csv("../../data/balanced_bcf_test.csv")

# drop unnecessary columns
train_df = train_df.drop(unused_cat_vars, axis=1)
train_df = train_df.drop(unused_con_vars, axis=1)
test_df = test_df.drop(unused_cat_vars, axis=1)
test_df = test_df.drop(unused_con_vars, axis=1)

# dummy variables
# select columns to encode as dummy variables
cols_to_encode = ['job', 'marital', 'education',
                  'housing', 'loan', 'contact', 'poutcome']

# encode columns as dummy variables
train_encoded_cols = pd.get_dummies(train_df[cols_to_encode])
test_encoded_cols = pd.get_dummies(test_df[cols_to_encode])

# replace original columns with dummy variable columns
train_df = pd.concat([train_df.drop(cols_to_encode, axis=1), 
                      train_encoded_cols], axis=1)
test_df = pd.concat([test_df.drop(cols_to_encode, axis=1), 
                      test_encoded_cols], axis=1)

# Move the churn column to the last.
train_df[response] = train_df.pop(response)
test_df[response] = test_df.pop(response)


# Scale the continous columns.
scaler = StandardScaler() # create StandardScaler object
train_df[con_vars] = scaler.fit_transform(train_df[con_vars])
test_df[con_vars] = scaler.fit_transform(test_df[con_vars])

# Split the data into training and testing sets
X_train = np.array(train_df.iloc[:, :-1].values)
y_train = np.array(train_df.iloc[:, -1].values)
X_test = np.array(test_df.iloc[:, :-1].values)
y_test = np.array(test_df.iloc[:, -1].values)


### Define the neural network architecture
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layers_stack = nn.Sequential(
            nn.Linear(43, 20),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layers_stack(x)
        return out

# Create an instance of the neural network
net = BinaryClassifier()

# Define the loss function (binary cross-entropy)
criterion = nn.BCELoss()

# Define the optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Convert the data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)


### Train the neural network
num_epochs = theepochtimes
batch_size = thebatchsize
num_batches = len(X_train) // batch_size

for epoch in range(num_epochs):
    for i in range(num_batches):
        # Get a batch of data
        start = i * batch_size
        end = (i + 1) * batch_size
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        # Forward pass
        outputs = net(X_batch)
        loss = criterion(outputs.squeeze(1), y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss for this epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

def plot_roc(actual_response, predicted_response):
    # calculate false positive rate, true positive rate,
    # and thresholds for different classification thresholds
    fpr, tpr, thresholds = roc_curve(actual_response,
                                     predicted_response)

    # plot ROC curve
    plt.plot(fpr, tpr)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

# Get the optimal metrics
def get_opt_accuracy(actual_response, predicted_response,
                     target_metric):
    nn_metrics = np.zeros((101, 7))
    seq = np.arange(0, 101)

    for i in seq:
        t = 0.01 * i
        # convert predicted probabilities to predicted labels
        y_pred_labels = (y_pred > t).float()

        # get confusion matrix
        cm = confusion_matrix(actual_response, y_pred_labels)

        if (cm[1, 0] + cm[1, 1]) > 0 \
          and (cm[0, 1] + cm[1, 1]) > 0 \
          and (cm[0, 0] + cm[0, 1]) > 0 \
          and (cm[0, 0] + cm[1, 0]):
            # calculate accuracy score
            accuracy = accuracy_score(actual_response, y_pred_labels)
            accuracy_check = (cm[0, 0] + cm[1, 1]) / (sum(sum(cm)))
            
            positive_precision = cm[1, 1] / (cm[1, 0] + cm[1, 1])
            positive_recall = cm[1, 1] / (cm[0, 1] + cm[1, 1])
            negative_precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            negative_recall = cm[0, 0] / (cm[0, 0] + cm[1, 0])

            new_row = [t, accuracy, accuracy_check, 
                             positive_precision, positive_recall, 
                             negative_precision, negative_recall]
            nn_metrics[i] = new_row
    
    column_names = ['threshold', 'accuracy', 'accuracy_check',
                    'positive_precision', 'positive_recall',
                    'negative_precision', 'negative_recall']
    nn_metrics_df = pd.DataFrame(nn_metrics, columns = column_names)
    max_row = nn_metrics_df['accuracy'].idxmax()

    return nn_metrics_df.iloc[max_row]

# Get metrics by set threshold
def get_accuracy_by_threshold(actual_response,
                              predicted_response, threshold):
    # convert predicted probabilities to predicted labels
    y_pred_labels = (y_pred > threshold).float()

    # get confusion matrix
    cm = confusion_matrix(actual_response, y_pred_labels)

    # calculate accuracy score
    accuracy = accuracy_score(actual_response, y_pred_labels)
    accuracy_check = (cm[0, 0] + cm[1, 1]) / (sum(sum(cm)))
    
    positive_precision = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    positive_recall = cm[1, 1] / (cm[0, 1] + cm[1, 1])
    negative_precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    negative_recall = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    
    nn_metrics = [threshold, accuracy_check, accuracy, 
                  positive_precision, positive_recall, 
                  negative_precision, negative_recall]

    return nn_metrics

# Evaluate the model on the test set
with torch.no_grad():
    # make predictions on the test set using your model
    y_pred = net(X_test)

    # calculate AUC score
    auc = roc_auc_score(y_test.numpy(), y_pred.numpy())

    # get optimal result
    nn_metric = get_opt_accuracy(y_test.numpy(),
                                 y_pred.numpy(),
                                 'accuracy')
    nn_metric_2 = get_accuracy_by_threshold(y_test.numpy(),
                                            y_pred.numpy(), 0.3)

print(f'Test AUC: {auc:.4f}\n\nTest Accuracy: \n{nn_metric}\n\n\
        Set threshold as 0.3:\n{nn_metric_2}')
