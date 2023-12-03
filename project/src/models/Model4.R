library(data.table)
library(reticulate)
library(xgboost)

# Load data
data <- fread('./project/volume/data/raw/kaggle_train.csv')
train_emb <- fread("./project/volume/data/raw/train_emb.csv")
test_emb <- fread("./project/volume/data/raw/test_emb.csv")

# Ensure 'reddit' is a factor
data$reddit <- as.factor(data$reddit)

# Perform PCA on training data
num_pca_components <- 11  # Set the number of principal components
pca_model <- prcomp(as.matrix(train_emb), center = TRUE, scale. = TRUE)
train_emb_pca <- predict(pca_model, as.matrix(train_emb))[, 1:num_pca_components]

# Save PCA model
saveRDS(pca_model, "./project/volume/models/pca_model.rds")

# Train XGBoost model on PCA dimensions
xgb_model_pca <- xgboost(
  data = as.matrix(train_emb_pca),
  label = data$reddit,
  nrounds = 10  # Adjust the number of boosting rounds
)

# Save XGBoost model
saveRDS(xgb_model_pca, "./project/volume/models/xgb_model_pca.rds")


# Apply PCA transformation to test data
test_emb_pca <- predict(pca_model, as.matrix(test_emb))[, 1:num_pca_components]
dtest <- xgb.DMatrix(as.matrix(test_emb_pca),missing=NA)

# Load the XGBoost model
loaded_xgb_model_pca <- readRDS("./project/volume/models/xgb_model_pca.rds")

# Make predictions on the test data
test_predictions_pca <- predict(loaded_xgb_model_pca, dtest)


test_predictions_pca
predicted_labels <- max.col(test_predictions_pca)
predicted_labels
