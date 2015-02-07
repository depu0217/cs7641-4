#
# Neural Network experiments for:
#     Credit Classifier
#
# Citations;
# Normalize() function from book by Brett Lantz. Some other code snippets may have been borrowed from this book as well:
#     Lantz, Brett. Machine Learning with R: Learn How to Use R to Apply Powerful Machine Learning Methods and Gain an Insight into Real-world Applications. 
#     Birmingham, UK: Packt Publishing, 2013.Original data from Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
#


library(nnet)
library(caret)
library(RCurl)
library(Metrics)
library(e1071)
library(gmodels)
library(devtools)
normalize <- function(x) { return((x - min(x)) / (max(x) - min(x)))}

# Read in credit data file and initialize training and test data frames
credit <- read.csv("credit-numeric.csv")
credit[2:20] <- as.data.frame(lapply(credit[2:20], normalize))

set.seed(1234)
credit <- credit[sample(nrow(credit)),]

#open output file
fname <- ("../output/Credit-NNET.Rout")
out <- file(fname, open="wt")
sink(out, type="output")
print("Begin testing with various training sizes")

for (i in seq(1:1)){
  #vary the size of the training partition based on the iteration; start with 5/10 of the data and go up to 9/10
  split <- floor(nrow(credit)*(4+i)/10)
  #always use the last 1/10 of the data for testing - use testBase as the starting index
  testBase <- floor(nrow(credit)*9/10)
  creditTrain <- credit[1:(split),2:17]
  creditTest <- credit[(testBase+1):nrow(credit),2:17]
  creditTrainLabels <- credit[1:split, 1]
  creditTestLabels <- credit[(testBase+1):nrow(credit), 1]
  
  # Test 1: Train once using the training data, then run against test data and check the error
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  neuralNetModel <- nnet(creditTrainLabels~., creditTrain,  size=13, maxit=i*1000, trace=T)
  print("Training Time: ")
  print(Sys.time()-startTime)
  print("Training Sample")
  pred <- predict(neuralNetModel, type="class", newdata=creditTrain)
  err <- ce(as.character(creditTrainLabels), pred)
  print(err, out)
  CrossTable(creditTrainLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual letter', 'predicted letter'))
  
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  print("Test Sample")
  pred <- predict(neuralNetModel, type="class", newdata=creditTest)
  print("Test Time: ")
  print(Sys.time()-startTime)
  err <- ce(as.character(creditTestLabels), pred)
  print(err, out)
  CrossTable(creditTestLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual letter', 'predicted letter'))
}

# Test 2: Train again. This time vary number of training iterations for the NNet and check for overfitting
#
print("Start Overfit Test")
for (i in seq(1:5)){
  split <- floor(nrow(credit)*9/10)
  testBase <- floor(nrow(credit)*9/10)
  creditTrain <- credit[1:(split),2:17]
  creditTest <- credit[(testBase+1):nrow(credit),2:17]
  creditTrainLabels <- credit[1:split, 1]
  creditTestLabels <- credit[(testBase+1):nrow(credit), 1]
  
  # Test 1: Train once using the training data, then run against test data and check the error
  neuralNetModel <- nnet(creditTrainLabels~., creditTrain,  size=13, maxit=(i*100), trace=T)
  print("Training Time: ")
  print(Sys.time()-startTime)
  print("Training Sample")
  pred <- predict(neuralNetModel, type="class", newdata=creditTrain)
  err <- ce(as.character(creditTrainLabels), pred)
  print(err, out)
  CrossTable(creditTrainLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual letter', 'predicted letter'))
  
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  print("Test Sample")
  pred <- predict(neuralNetModel, type="class", newdata=creditTest)
  print("Test Time: ")
  print(Sys.time()-startTime)
  err <- ce(as.character(creditTestLabels), pred)
  print(err, out)
  CrossTable(creditTestLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual letter', 'predicted letter'))
}


# Test 3: Train again. Do k-fold cross-validation
#
print("Start Cross-Validation")
totalError <- c()
cv <- 9
cvDivider <- floor(nrow(credit) /(cv+1))
cvDivider
nrow(credit)
totalTrainingError <-c()
totalTestError <-c()

for (cv in seq(1:cv)) {
  dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider-1))
  dataTrain <- credit[-dataTestIndex, 2:17]
  dataTest <- credit[dataTestIndex, 2:17]
  dataTrainLabels <- credit[-dataTestIndex, 1]
  dataTestLabels <- credit[dataTestIndex, 1]
  
  neuralNetModel <- nnet(dataTrainLabels~., dataTrain,  size=13, maxit=500, trace=T)
  print("Test Sample")
  pred <- predict(neuralNetModel, type="class", newdata=dataTest)
  err2 <- ce(as.character(dataTestLabels), pred)
  print(err, out)
  totalTestError <- c(totalTestError, err2)
}
print("test error")
print(totalTestError)
print(mean(totalTestError))
print(sd(totalTestError))
print(CrossTable(dataTestLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual default', 'predicted default')), out)

closeAllConnections()
sink()
print("Done!")