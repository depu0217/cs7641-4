# This is the letters.R file. It runs the letterdata.csv file through all Supervised Learning algorithms#
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
library(grid)
library(MASS)
library(nnet)
library(class)
library(kernlab)
library(caret)
normalize <- function(x) { return((x - min(x)) / (max(x) - min(x)))}

# Read in letterdata file and randomize 
letters <- read.csv("letterdata.csv")
letters[2:17] <- as.data.frame(lapply(letters[2:17], normalize))
set.seed(1234)
letters <- letters[sample(nrow(letters)),]

#open output file
fname <- ("../output/Letters-Tree.Rout")
out <- file(fname, open="wt")
sink(out, type="output")

for (i in seq(1:5)){
  #vary the size of the training partition based on the iteration; start with 5/10 of the data and go up to 9/10
  split <- floor(nrow(letters)*(4+i)/10)
  #always use the last 1/10 of the data for testing - use testBase as the starting index
  testBase <- floor(nrow(letters)*9/10)
  lettersTrain <- letters[1:(split),2:17]
  lettersTest <- letters[(testBase+1):nrow(letters),2:17]
  lettersTrainLabels <- letters[1:split, 1]
  lettersTestLabels <- letters[(testBase+1):nrow(letters), 1]
  
  #prop.table(table(lettersTrain$letter))
  #prop.table(table(lettersTest$letter))
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  treeModel <- C5.0(lettersTrain, lettersTrainLabels)
  print("Training Time: ")
  print(Sys.time()-startTime)
  pred <- predict(treeModel, lettersTrain)
  print("Training Accuracy", out)
  print(postResample(lettersTrainLabels,pred), out)
  print("Tree Size:")
  print(treeModel$size)
  #print(summary(treeModel), out)
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  pred <- predict(treeModel, lettersTest)
  print("Test Time: ")
  print(Sys.time()-startTime)
  print("Test Accuracy", out)
  print(postResample(lettersTestLabels,pred), out)
  #CrossTable(lettersTestLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual letter', 'predicted letter'))
}

print("NO GLOBAL PRUNING")
for (i in seq(1:5)){
  #vary the size of the training partition based on the iteration; start with 5/10 of the data and go up to 9/10
  split <- floor(nrow(letters)*(4+i)/10)
  #always use the last 1/10 of the data for testing - use testBase as the starting index
  testBase <- floor(nrow(letters)*9/10)
  lettersTrain <- letters[1:(split),2:17]
  lettersTest <- letters[(testBase+1):nrow(letters),2:17]
  lettersTrainLabels <- letters[1:split, 1]
  lettersTestLabels <- letters[(testBase+1):nrow(letters), 1]
  
  #prop.table(table(lettersTrain$letter))
  #prop.table(table(lettersTest$letter))
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  treeModel <- C5.0(lettersTrain, lettersTrainLabels, noGlobalPruning=TRUE)
  print("Training Time: ")
  print(Sys.time()-startTime)
  pred <- predict(treeModel, lettersTrain)
  print("Training Accuracy", out)
  print(postResample(lettersTrainLabels,pred), out)
  print("Tree Size:")
  print(treeModel$size)
  #print(summary(treeModel), out)
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  pred <- predict(treeModel, lettersTest)
  print("Test Time: ")
  print(Sys.time()-startTime)
  print("Test Accuracy", out)
  print(postResample(lettersTestLabels,pred), out)
  #CrossTable(lettersTestLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual letter', 'predicted letter'))
}

closeAllConnections()
sink()