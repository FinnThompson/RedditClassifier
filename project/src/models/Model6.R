# Step 1 - Install the necessary libraries
install.packages('xgboost')  # for fitting the xgboost model
install.packages('caret')    # for general data preparation and model fitting
install.packages('e1071')
library(xgboost)
library(caret)
library(e1071)
library(data.table)

# Step 2 - Read the datasets and explore the data
data <- fread('./project/volume/data/raw/kaggle_train.csv')
train_emb <- fread("./project/volume/data/raw/train_emb.csv")
test_emb <- fread("./project/volume/data/raw/test_emb.csv")

# Assuming your embeddings are already in train_emb and test_emb
#train_emb <- as.matrix(train_emb)
#test_emb <- as.matrix(test_emb)

head(data)
dim(train_emb)

# Step 3 - Train and Test data
X_train = train_emb                                # independent variables for train
y_train = data$reddit                              # dependent variables for train

X_test = test_emb                                  # independent variables for test
# y_test is not available for the test set in the original data
# Replace this with the actual variable if it is available in your context

# Convert the train and test data into xgboost matrix type.
xgboost_train = xgb.DMatrix(data = X_train, label = as.numeric(y_train))
xgboost_test = xgb.DMatrix(data = X_test)






example_sub <- fread("./project/volume/data/raw/example_sub.csv")

