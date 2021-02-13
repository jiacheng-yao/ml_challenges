library("yardstick")
library("tidyverse")
library("caret")
library("xgboost")
library("logging")
df_final <- read.csv("~/Codes/simplyrational/app_final(w_target).csv", header = TRUE)
df_final$TARGET <- factor(df_final$TARGET, labels=c("Others", "With Difficulties"))

#Create 10 equally size folds
folds <- cut(seq(1,nrow(df_final)),breaks=10,labels=FALSE)
opt_params <- list(booster = "gbtree", 
               objective = "binary:logistic", 
               eta=0.3, 
               gamma=0.01, 
               max_depth=2, 
               min_child_weight=1, 
               subsample=0.723, 
               colsample_bytree=0.742)

#Perform 10 fold cross validation

bacc_agg_xgb <- 0
for(i in 1:10){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- df_final[testIndexes, ]
  trainData <- df_final[-testIndexes, ]
  
  #XGBoost
  model <- train(TARGET ~., data = trainData, method = "xgbTree")
  
  pred_label <- model %>% predict(testData)
  bacc_tmp <- bal_accuracy_vec(testData$TARGET, pred_label)
  bacc_agg_xgb <- bacc_agg_xgb + bacc_tmp
  loginfo(paste("BACC of current fold: ", format(bacc_tmp, nsmall = 5)), sep="")
}
bacc_agg_xgb <- bacc_agg_xgb/10.0
loginfo(paste("Average BACC: ", format(bacc_agg_xgb, nsmall = 5)), sep="")