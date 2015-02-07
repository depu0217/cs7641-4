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
normalize <- function(x) { return((x - min(x)) / (max(x) - min(x)))}

# Read in letters data file. Then randomize and initialize training and test data frames
# Split between training and test is 80/20
#
letters <- read.csv("letterdata.csv")
letters[2:17] <- as.data.frame(lapply(letters[2:17], normalize))
table(letters$letter)
set.seed(1234)
letters <- letters[sample(nrow(letters)),]

#open output file
fname <- ("../output/Letters-NNET.Rout")
out <- file(fname, open="wt")
sink(out, type="output")

print("Start Training Set Size Test")
for (i in seq(1:5)){
#vary the size of the training partition based on the iteration; start with 5/10 of the data and go up to 9/10
  split <- floor(nrow(letters)*(4+i)/10)
  #always use the last 1/10 of the data for testing - use testBase as the starting index
  testBase <- floor(nrow(letters)*9/10)
  lettersTrain <- letters[1:(split),2:17]
  lettersTest <- letters[(testBase+1):nrow(letters),2:17]
  lettersTrainLabels <- letters[1:split, 1]
  lettersTestLabels <- letters[(testBase+1):nrow(letters), 1]

  startTime<-as.POSIXlt(Sys.time(), "UTC")
  # Test 1: Train once using the training data, then run against test data and check the error
  neuralNetModel <- nnet(lettersTrainLabels~., lettersTrain,  size=1, maxit=500, trace=T)
  print("Training Time: ")
  print(Sys.time()-startTime)
  print("Training Sample")
  pred <- predict(neuralNetModel, type="class", newdata=lettersTrain)
  err <- ce(as.character(lettersTrainLabels), pred)
  print(err, out)
  # CrossTable(lettersTrainLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual letter', 'predicted letter'))
  
  print("Test Sample")
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  pred <- predict(neuralNetModel, type="class", newdata=lettersTest)
  print("Test Time: ")
  print(Sys.time()-startTime)
  err <- ce(as.character(lettersTestLabels), pred)
  print(err, out)
  # CrossTable(lettersTestLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual letter', 'predicted letter'))
}


# Test 2: Train again. This time vary number of training iterations for the NNet and check for overfitting
#
print("Start Overfit Test")
for (i in seq(1:5)){
  split <- floor(nrow(letters)*9/10)
  testBase <- floor(nrow(letters)*9/10)
  lettersTrain <- letters[1:(split),2:17]
  lettersTest <- letters[(testBase+1):nrow(letters),2:17]
  lettersTrainLabels <- letters[1:split, 1]
  lettersTestLabels <- letters[(testBase+1):nrow(letters), 1]
  
  # Test 1: Train once using the training data, then run against test data and check the error
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  neuralNetModel <- nnet(lettersTrainLabels~., lettersTrain,  size=1, maxit=(i*100), trace=T)
  pred <- predict(neuralNetModel, type="class", newdata=lettersTrain)
  print("Training Time: ")
  print(Sys.time()-startTime)
  err <- ce(as.character(lettersTrainLabels), pred)
  print("Training Sample")
  print(err, out)
  # CrossTable(lettersTrainLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual letter', 'predicted letter'))
  
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  pred <- predict(neuralNetModel, type="class", newdata=lettersTest)
  print("Test Time: ")
  print(Sys.time()-startTime)
  err <- ce(as.character(lettersTestLabels), pred)
  print("Test Sample")
  print(err, out)
  # CrossTable(lettersTestLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual letter', 'predicted letter'))
}

closeAllConnections()
sink()
print("Done!")