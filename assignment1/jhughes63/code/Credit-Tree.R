# This is the Credit-Tree.R file. It runs the credit.csv file through the Tree Supervised Learning algorithm
# Initial conditions:
#    All packages listed in the 'libraries' section of the code must be installed prior to running this script
#    
# Citations;
# Normalize() function from book by Brett Lantz. Some other code snippets may have been borrowed from this book as well:
#     Lantz, Brett. Machine Learning with R: Learn How to Use R to Apply Powerful Machine Learning Methods and Gain an Insight into Real-world Applications. 
#     Birmingham, UK: Packt Publishing, 2013.Original data from Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
#

library(C50)
library(gmodels)
library(neuralnet)
library(nnet)
library(class)
library(kernlab)
library(caret)
normalize <- function(x) { return((x - min(x)) / (max(x) - min(x)))}

# Read in credit data file and randomize 
credit <- read.csv("credit.csv")
set.seed(1234)
credit <- credit[sample(nrow(credit)),]

#open output file
fname <- ("../output/Credit-Tree.Rout")
out <- file(fname, open="wt")
sink(out, type="output")

for (i in seq(1:5)){
#vary the size of the training partition based on the iteration; start with 5/10 of the data and go up to 9/10
  split <- floor(nrow(credit)*(4+i)/10)
  #always use the last 1/10 of the data for testing - use testBase as the starting index
  testBase <- floor(nrow(credit)*9/10)
  creditTrain <- credit[1:(split),2:15]
  creditTest <- credit[(testBase+1):nrow(credit),2:15]
  creditTrainLabels <- credit[1:split, 1]
  creditTestLabels <- credit[(testBase+1):nrow(credit), 1]
  
  # Test Run #1; Train, predict, check results
  # Decision Tree with Pruning; no boosting
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  treeModel <- C5.0(creditTrain, creditTrainLabels)
  print("Training Time: ")
  print(Sys.time()-startTime)
  print("Tree Size:")
  print(treeModel$size)
  print(summary(treeModel))
  pred <- predict(treeModel, creditTrain)
  #print(CrossTable(creditTrainLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual default', 'predicted default')),out)
  #print(summary(treeModel), out)
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  pred <- predict(treeModel, creditTest)
  print("Test Time: ")
  print(Sys.time()-startTime)
  #print(postResample(creditTestLabels,pred), out)
  print(CrossTable(creditTestLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual default', 'predicted default')),out)
}

print("NO GLOBAL PRUNING")
for (i in seq(1:5)){
  #vary the size of the training partition based on the iteration; start with 5/10 of the data and go up to 9/10
  split <- floor(nrow(credit)*(4+i)/10)
  #always use the last 1/10 of the data for testing - use testBase as the starting index
  testBase <- floor(nrow(credit)*9/10)
  creditTrain <- credit[1:(split),2:15]
  creditTest <- credit[(testBase+1):nrow(credit),2:15]
  creditTrainLabels <- credit[1:split, 1]
  creditTestLabels <- credit[(testBase+1):nrow(credit), 1]
  
  # Test Run #1; Train, predict, check results
  # Decision Tree with Pruning; no boosting
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  treeModel <- C5.0(creditTrain, creditTrainLabels, noGlobalPruning=TRUE)
  print("Training Time: ")
  print(Sys.time()-startTime)
  print("Tree Size:")
  print(treeModel$size)
  pred <- predict(treeModel, creditTrain)
  print(CrossTable(creditTrainLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual default', 'predicted default')),out)
  #print(summary(treeModel), out)
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  pred <- predict(treeModel, creditTest)
  print("Test Time: ")
  print(Sys.time()-startTime)
  #print(postResample(creditTestLabels,pred), out)
  print(CrossTable(creditTestLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual default', 'predicted default')),out)
}

closeAllConnections()
sink()