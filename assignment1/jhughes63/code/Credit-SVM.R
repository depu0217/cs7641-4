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
normalize <- function(x) { return((x - min(x)) / (max(x) - min(x)))}

# Read in credit data file and initialize training and test data frames
credit <- read.csv("credit-numeric.csv")
#credit[2:20] <- data.frame(lapply(credit[2:20], as.numeric))
credit[2:20] <- as.data.frame(lapply(credit[2:20], normalize))
head(credit)
set.seed(1234)
credit <- credit[sample(nrow(credit)),]

#open output file
fname <- ("../output/Credit-SVM.Rout")
out <- file(fname, open="wt")
sink(out, type="output")
print("Begin testing with various training sizes")

for (i in seq(1:5)){
  #vary the size of the training partition based on the iteration; start with 5/10 of the data and go up to 9/10
  split <- floor(nrow(credit)*(4+i)/10)
  #always use the last 1/10 of the data for testing - use testBase as the starting index
  testBase <- floor(nrow(credit)*9/10)
  creditTrain <- credit[1:(split),]
  creditTest <- credit[(testBase+1):nrow(credit),]
  creditTrainLabels <- credit[1:split, 1]
  creditTestLabels <- credit[(testBase+1):nrow(credit), 1]
  
  # Test Run; Train, predict, check results
  # Neural Network - input data is normalized earlier in this routine
  #
  print("interation:")
  print(i)
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  svmModel <- ksvm(default ~ ., data = creditTrain, kernel = "vanilladot")
  print("Train Time: ")
  print(Sys.time()-startTime)
  print("Train Prediction")
  pred <- predict(svmModel, creditTrain)
  print(CrossTable(creditTrainLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual default', 'predicted default')),out)
  
  startTime<-as.POSIXlt(Sys.time(), "UTC")
  pred <- predict(svmModel, creditTest)
  print("Test Time: ")
  print(Sys.time()-startTime)
  print(CrossTable(creditTestLabels, pred, prop.chisq=FALSE, prop.r=FALSE, dnn=c('actual default', 'predicted default')),out)
    
  print(table(pred, creditTestLabels), out)
  agreement <- pred == creditTestLabels
  table(agreement)
  prop.table(table(agreement))
}
closeAllConnections()
sink()
