library(httr)
library(data.table)
library(Rtsne)
library(ggplot2)
library(reticulate)


data<-fread('./project/volume/data/raw/kaggle_train.csv')
train_emb <- fread("./project/volume/data/raw/train_emb.csv")
test_emb <- fread("./project/volume/data/raw/test_emb.csv")

tsne<-Rtsne(train_emb,perplexity=5)

tsne_dt<-data.table(tsne$Y)

tsne_dt$reddit<-data$reddit
tsne_dt$id<-data$id

ggplot(tsne_dt,aes(x=V1,y=V2,col=reddit,label=id))+geom_text()

# Perform PCA
pca_model <- prcomp(train_emb)

# Save PCA model
saveRDS(pca_model, "./project/volume/models/pca_model.rds")


# Ensure 'reddit' is a factor
data$reddit <- as.factor(data$reddit)

# Train XGBoost model
xgb_model <- train(
  x = as.matrix(train_emb),
  y = data$reddit,
  method = "xgbTree",
  trControl = ctrl,
  verbose = FALSE
)

# Save the model
saveRDS(xgb_model, "./project/volume/models/xgb_model.rds")




pca.model<-saveRDS("./project/volume/models/pca.model")
xgb.model<-saveRDS("./project/volume/models/xgb.model")






