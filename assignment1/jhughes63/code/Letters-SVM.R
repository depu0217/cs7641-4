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
normalize <- function(x) { return((x - min(x)) / (max(x) - min(x)))}

# Read in letters data file. Then randomize and initialize training and test data frames
# Split between training and test is 80/20
letters <- read.csv("letterdata.csv")
letters[2:17] <- as.data.frame(lapply(letters[2:17], normalize))
table(letters$letter)
set.seed(1234)
letters <- letters[sample(nrow(letters)),]

#open output file
fname <- ("../output/Letters-SVM.Rout")
out <- file(fname, open="wt")
sink(out, type="output")

for (i in seq(1:5)){
  #vary the size of the training partition based on the iteration; start with 5/10 of the data and go up to 9/10
  split <- floor(nrow(letters)*(4+i)/10)
  #always use the last 1/10 of the data for testing - use testBase as the starting index
  testBase <- floor(nrow(letters)*9/10)
  lettersTrain <- letters[1:(split),]
  lettersTest <- letters[(testBase+1):nrow(letters),]
  lettersTrainLabels <- letters[1:split, 1]
  lettersTestLabels <- letters[(testBase+1):nrow(letters), 1]

  # Test Run X; Train, predict, check results
  # SVM 
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  svmModel <- ksvm(letter ~ ., data = lettersTrain, kernel = "rbfdot")
  print("Training Time: ")
  print(Sys.time()-startTime)
  pred <- predict(svmModel, lettersTrain)
  print("Training Accuracy", out)
  print(table(pred, lettersTrainLabels), out)
  agreement <- pred == lettersTrainLabels
  print(table(agreement), out)
  print(prop.table(table(agreement)), out)
  # print(summary(svmModel), out)
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  pred <- predict(svmModel, lettersTest)
  print("Test Time: ")
  print(Sys.time()-startTime)
  print("Test Accuracy", out)
  print(table(pred, lettersTestLabels), out)
  agreement <- pred == lettersTestLabels
  print(table(agreement), out)
  print(prop.table(table(agreement)), out)
}
#print(CrossTable(dataTest$letter, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual letter', 'predicted letter')), out)
closeAllConnections()
sink()