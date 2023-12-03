library(data.table)

train_emb <- fread("./project/volume/data/raw/train_emb.csv")
test_emb <- fread("./project/volume/data/raw/test_emb.csv")


DogData$breed <- NA
DogData$breed[c(1, 5, 6)] <- c(3, 2, 4)
head(example_sub)
fwrite(DogData, "./project/volume/data/interim/finalData.csv")

