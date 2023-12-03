library(data.table)
library(Rtsne)
library(ggplot2)
library(caret)
library(ggplot2)
library(ClusterR)

set.seed(3)

# load in data 
train_emb <- fread("./project/volume/data/raw/train_emb.csv")
test_emb <- fread("./project/volume/data/raw/test_emb.csv")
example_sub <- fread("./project/volume/data/raw/example_sub.csv")

#breed<-DogData$breed
#DogData$breed<-NULL


#ids <- example_sub$id

#drops<- c('id')
#DogData<-DogData[, !drops, with = FALSE]


# do a pca
pca<-prcomp(train_emb)
#pca_test<-prcomp(test_emb)

# look at the percent variance explained by each pca
screeplot(pca)


# look at the rotation of the variables on the PCs
pca




# use the unclass() function to get the data in PCA space
pca_dt<-data.table(unclass(pca)$x)


# run t-sne on the PCAs, note that if you already have PCAs you need to set pca=F or it will run a pca again. 
# pca is built into Rtsne, ive run it seperatly for you to see the internal steps

#tsne<-Rtsne(pca_dt,pca=T,perplexity=5,check_duplicates = F)
tsne<-Rtsne(train_emb,pca=T,perplexity=30,check_duplicates = F)

tsne_test<-Rtsne(test_emb,pca=T,perplexity = 5, check_duplicates = F)


# grab out the coordinates
tsne_dt<-data.table(tsne$Y)


ggplot(tsne_dt,aes(x=V1,y=V2))+geom_point()



# use a gaussian mixture model to find optimal k and then get probability of membership for each row to each group

# this fits a gmm to the data for all k=1 to k= max_clusters, we then look for a major change in likelihood between k values
k_bic<-Optimal_Clusters_GMM(tsne_dt[,.(V1,V2)],max_clusters = 30,criterion = "BIC")
k_bic
# now we will look at the change in model fit between successive k values
delta_k<-c(NA,k_bic[-1] - k_bic[-length(k_bic)])


# I'm going to make a plot so you can see the values, this part isnt necessary
del_k_tab<-data.table(delta_k=delta_k,k=1:length(delta_k))

# plot 
ggplot(del_k_tab,aes(x=k,y=-delta_k))+geom_point()+geom_line()+
  scale_x_continuous(breaks = scales::pretty_breaks(n = 10))+
  geom_text(aes(label=k),hjust=0, vjust=-1)



opt_k <- 11

saveRDS(k_bic,"./project/volume/models/k_bic_model")
# now we run the model with our chosen k value

gmm_data<-GMM(tsne_dt[,.(V1,V2)],opt_k)
gmm_data$Log_likelihood

# the model gives a log-likelihood for each datapoint's membership to each cluster, me need to convert this 
# log-likelihood into a probability

l_clust<-gmm_data$Log_likelihood^10

l_clust<-data.table(l_clust)
l_clust

net_lh<-apply(l_clust,1,FUN=function(x){sum(1/x)})

cluster_prob<-1/l_clust/net_lh

cluster_prob

# Print the probabilities
print(cluster_prob)

setnames(cluster_prob, "V1", "redditscience")
setnames(cluster_prob, "V2", "redditCFB")
setnames(cluster_prob, "V3", "redditCooking")
setnames(cluster_prob, "V4", "reddittravel")
setnames(cluster_prob, "V5", "redditStockMarket")
setnames(cluster_prob, "V6", "redditRealEstate")
setnames(cluster_prob, "V7", "redditmagicTCG")
setnames(cluster_prob, "V8", "redditMachineLearning")
setnames(cluster_prob, "V9", "redditvideogames")
setnames(cluster_prob, "V10", "redditcars")
setnames(cluster_prob, "V11", "redditpolitics")


l_clust_probs


final_data <- cbind(id = example_sub$id, 
                    redditcars = cluster_prob$redditcars, 
                    redditCFB = cluster_prob$redditCFB, 
                    redditCooking = cluster_prob$redditCooking, 
                    redditMachineLearning = cluster_prob$redditMachineLearning,
                    redditmagicTCG = cluster_prob$redditmagicTCG,
                    redditpolitics = cluster_prob$redditpolitics,
                    redditRealEstate = cluster_prob$redditRealEstate,
                    redditscience = cluster_prob$redditscience,
                    redditStockMarket = cluster_prob$redditStockMarket,
                    reddittravel = cluster_prob$reddittravel,
                    redditvideogames = cluster_prob$redditvideogames)
final_data


fwrite(final_data, "./project/volume/data/external/Submission1.csv")

