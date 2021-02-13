library("FFTrees")
library("yardstick")
library("logging")

df_final <- read.csv("~/Codes/simplyrational/app_final(w_target).csv", header = TRUE)
df_final$TARGET <- as.logical(df_final$TARGET)

#Create 10 equally size folds
folds <- cut(seq(1,nrow(df_final)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation
bacc_agg <- 0

for(i in 1:10){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- df_final[testIndexes, ]
  trainData <- df_final[-testIndexes, ]
  
  #FFT
  credit_FFT <- FFTrees(formula = TARGET ~ .,               # The variable we are predicting
                        data = trainData,                    # Training data
                        data.test = testData,                # Testing data
                        main = "Credit Decision",                # Main label
                        decision.labels = c("Others", "With Difficulties"), 
                        do.comp = FALSE) # Label for decisions
  
  pred_label <- predict(credit_FFT, testData)
  
  truth_factor <- factor(testData$TARGET, labels=c("Others", "With Difficulties"))
  pred_factor <- factor(pred_label, labels=c("Others", "With Difficulties"))
  bacc_tmp <- bal_accuracy_vec(truth_factor, pred_factor)
  bacc_agg <- bacc_agg + bacc_tmp
  loginfo(paste("BACC of current fold: ", format(bacc_tmp, nsmall = 5)), sep="")
}
bacc_agg <- bacc_agg/10.0
loginfo(paste("Average BACC: ", format(bacc_agg, nsmall = 5)), sep="")