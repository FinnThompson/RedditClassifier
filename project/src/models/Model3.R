install.packages("MLmetrics")

library(httr)
library(data.table)
library(Rtsne)
library(ggplot2)
library(reticulate)
library(caret)
library(xgboost)
library(MLmetrics)

# Load data
data <- fread('./project/volume/data/raw/kaggle_train.csv')
train_emb <- fread("./project/volume/data/raw/train_emb.csv")
test_emb <- fread("./project/volume/data/raw/test_emb.csv")

# Ensure 'reddit' is a factor
data$reddit <- as.factor(data$reddit)

# Perform t-SNE
tsne <- Rtsne(train_emb, perplexity = 5)
tsne_dt <- data.table(tsne$Y)
tsne_dt$reddit <- data$reddit
tsne_dt$id <- data$id

# Save the t-SNE model
saveRDS(tsne, "./project/volume/models/tsne_model.rds")

# Visualize t-SNE
ggplot(tsne_dt, aes(x = V1, y = V2, col = reddit, label = id)) +
  geom_text()

xgb_model_tsne <- xgboost(
  data = as.matrix(tsne_dt[, .(V1, V2)]),
  label = tsne_dt$reddit,
  nrounds = 1000  # Adjust the number of boosting rounds
)

saveRDS(xgb_model_tsne, "./project/volume/models/xgb_model_tsne.rds")


loaded_tsne <- readRDS("./project/volume/models/tsne_model.rds")

# Apply t-SNE transformation to new data
new_data_tsne <- predict(loaded_tsne, as.matrix(test_emb))

# Load the XGBoost model
loaded_xgb_model_tsne <- readRDS("./project/volume/models/xgb_model_tsne.rds")

# Make predictions on the new data
new_data_predictions_tsne <- predict(loaded_xgb_model_tsne, new_data_tsne)










train_pca <- predict(pca_model, newdata=as.data.frame(train_emb))
train_pca
train_pca <- data.table(train_pca)
train_pca<-train_pca[,.(PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,PC11)]


# Set up train control for multi-class classification
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = multiClassSummary)

# Train XGBoost model
xgb_model <- train(
  x = as.matrix(train_pca),
  y = data$reddit,
  method = "xgbTree",
  trControl = ctrl,
  verbose = FALSE
)

# Save the model
saveRDS(xgb_model, "./project/volume/models/xgb_model.rds")

# Perform PCA
pca_model <- prcomp(train_emb)
k_bic<-Optimal_Clusters_GMM(tsne_dt[,.(V1,V2)],max_clusters = 30,criterion = "BIC")

# Save PCA model
saveRDS(k_bic, "./project/volume/models/pca_model.rds")


# Loading the models (if needed)
xgb_model <- readRDS("./project/volume/models/xgb_model.rds")
pca_model <- readRDS("./project/volume/models/pca_model.rds")

test_pca <- predict(pca_model, newdata=as.data.frame(test_emb))
test_pca
test_pca <- data.table(test_pca)
test_pca
test_pca<-test_pca[,.(PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,PC11)]

dtest <- xgb.DMatrix(as.matrix(test_pca),missing=NA)
test_pca
pred<-predict(xgb_model, newdata = dtest)



eval_new_text<-function(new_text){
  new_text_emb<-getEmbeddings(new_text)
  
  new_text_pca<-predict(pca.model,newdata=as.data.frame(new_text_emb))
  
  new_text_pca<-data.table(new_text_pca)
  
  new_text_pca<-new_text_pca[,.(PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10)]
  
  dtest <- xgb.DMatrix(as.matrix(new_text_pca),missing=NA)
  
  pred<-predict(xgb.model, newdata = dtest )
  
  results<-matrix(pred,ncol=11,byrow=T)
  results<-data.table(results)
  
  output<-ex_sub[1,2:ncol(ex_sub)]
  
  output[1,]<-results
  
  output$id<-1
  
  output<-melt(output,id.vars = "id",variable.name = "reddit", value.name = "probability")
  
  output<-output[order(-probability)]
  
  output$id<-NULL
  
  print(output)
}

eval_new_text("How much would you pay for a 2020 model toyota tundra with the base package out the door?")
