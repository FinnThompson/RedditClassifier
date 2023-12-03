library(xgboost)
library(data.table)
library(caret)
library(Metrics)


data <- fread('./project/volume/data/raw/kaggle_train.csv')
train_emb <- fread("./project/volume/data/raw/train_emb.csv")
test_emb <- fread("./project/volume/data/raw/test_emb.csv")
example_sub <- fread("./project/volume/data/raw/example_sub.csv")

# Ensure 'reddit' is a factor
data$reddit <- as.factor(data$reddit)
data$reddit

train_labels

label_dict <- levels(data$reddit)

# Train data
train_matrix <- as.matrix(train_emb)
train_labels <- as.numeric(data$reddit)

train_labels <- as.integer(factor(train_labels)) - 1
nlevels(as.factor(train_labels))
train_labels

# Test data
test_matrix <- as.matrix(test_emb)


param <- list(  objective           = "reg:squarederror",
                gamma               =0.00,
                booster             = "gbtree",
                eval_metric         = "rmse",
                eta                 = 0.015,
                max_depth           = 5,
                alpha = 0.9,
                min_child_weight    = 10,
                subsample           = 0.9,
                colsample_bytree    = 0.55,
                tree_method = 'hist'
                
)



# notice that I've removed label=departure delay in the dtest line, I have departure delay available to me with the in my dataset but
# you dont have price for the house prices.

set.seed(7)


# Train XGBoost model
xgb_model <- xgboost(data = train_matrix, max_depth = 5, alpha = 0, label = train_labels, nrounds = 100, objective = "multi:softprob", num_class = nlevels(as.factor(train_labels)))

# Make predictions on test data
test_probs <- predict(xgb_model, test_matrix, type = "prob")

test_probs
results <- matrix(test_probs, ncol = 11, byrow=T)

# Convert the predicted probabilities to a data frame
#result_df <- as.data.frame(test_probs)


# Convert the transposed matrix to a data frame
#result_df <- as.data.frame(test_probs)

# Print the resulting data frame
#print(result_df)


#results<-matrix(result_df,ncol=11,byrow=T)
#results

results <- as.data.frame(results)
results

setnames(results, "V1", "redditcars")
setnames(results, "V2", "redditCFB")
setnames(results, "V3", "redditCooking")
setnames(results, "V4", "redditMachineLearning")
setnames(results, "V5", "redditmagicTCG")
setnames(results, "V6", "redditpolitics")
setnames(results, "V7", "redditRealEstate")
setnames(results, "V8", "redditscience")
setnames(results, "V9", "redditStockMarket")
setnames(results, "V10", "reddittravel")
setnames(results, "V11", "redditvideogames")


final_data <- cbind(id = example_sub$id, 
                    redditcars = results$redditcars, 
                    redditCFB = results$redditCFB, 
                    redditCooking = results$redditCooking, 
                    redditMachineLearning = results$redditMachineLearning,
                    redditmagicTCG = results$redditmagicTCG,
                    redditpolitics = results$redditpolitics,
                    redditRealEstate = results$redditRealEstate,
                    redditscience = results$redditscience,
                    redditStockMarket = results$redditStockMarket,
                    reddittravel = results$reddittravel,
                    redditvideogames = results$redditvideogames)

final_data


fwrite(final_data, "./project/volume/data/external/Submission2.csv")
