## We will use packeges: caret, earth, kernlab, and nnet
install.packages(c("caret", "earth", "kernlab", "nnet"))

set.seed(1)
x <- runif(100, min = 2, max = 10)
y <- sin(x) + rnorm(length(x)) * .25
sinData <- data.frame(x = x, y = y)
plot(x, y)

## Create a grid of x values to use for prediction

dataGrid <- data.frame(x = seq(2, 10, length = 100))


library(kernlab)
rbfSVM <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = "automatic",
               C = 1, epsilon = 0.1)
modelPrediction <- predict(rbfSVM, newdata = dataGrid)
plot(x,y)
points(x = dataGrid$x, y = modelPrediction[,1], type = "l", col = "blue")


rbfSVM2 <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = "automatic",
               C = 1, epsilon = 0.2)
modelPrediction2 <- predict(rbfSVM2, newdata = dataGrid)
plot(x,y)
points(x = dataGrid$x, y = modelPrediction2[,1], type = "l", col = "blue")


rbfSVM3 <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = "automatic",
               C = 2, epsilon = 0.1)
modelPrediction3 <- predict(rbfSVM3, newdata = dataGrid)
plot(x,y)
points(x = dataGrid$x, y = modelPrediction3[,1], type = "l", col = "blue")


rbfSVM4 <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = "automatic",
                C = 2, epsilon = 0.2)
modelPrediction4 <- predict(rbfSVM4, newdata = dataGrid)
plot(x,y)
points(x = dataGrid$x, y = modelPrediction4[,1], type = "l", col = "blue")

rbfSVM5 <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = "automatic",
                C = 2, epsilon = 0.2)
modelPrediction5 <- predict(rbfSVM5, newdata = dataGrid)
plot(x,y)
points(x = dataGrid$x, y = modelPrediction5[,1], type = "l", col = "blue")



#7.1(b)

rbfSVM6 <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = list(sigma = 0.5), C = 2, epsilon = 0.2)
modelPrediction6 <- predict(rbfSVM6, newdata = dataGrid)
plot(x, y)
points(x = dataGrid$x, y = modelPrediction6[,1],type = "l", col = "blue")


rbfSVM7 <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = list(sigma = 1), C = 2, epsilon = 0.2)
modelPrediction7 <- predict(rbfSVM7, newdata = dataGrid)
plot(x, y)
points(x = dataGrid$x, y = modelPrediction7[,1],type = "l", col = "blue")

rbfSVM8 <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = list(sigma = 3), C = 2, epsilon = 0.2)
modelPrediction8 <- predict(rbfSVM8, newdata = dataGrid)
plot(x, y)
points(x = dataGrid$x, y = modelPrediction8[,1],type = "l", col = "blue")

rbfSVM9 <- ksvm(x = x, y = y, data = sinData, kernel ="rbfdot", kpar = list(sigma = 8), C = 2, epsilon = 0.2)
modelPrediction8 <- predict(rbfSVM9, newdata = dataGrid)
plot(x, y)
points(x = dataGrid$x, y = modelPrediction9[,1],type = "l", col = "blue")



#7.2 (a)


library(mlbench)
library(caret)
set.seed(200)
trainingData <- mlbench.friedman1(200, sd = 1)
trainingData$x <- data.frame(trainingData$x)
featurePlot(trainingData$x, trainingData$y)


##knn model

knnModel <- train(x = trainingData$x,y = trainingData$y,method = "knn",preProc = c("center", "scale"), tuneLength = 10)
knnModel
knnPred <- predict(knnModel, newdata = testData$x)
postResample(pred = knnPred, obs = testData$y)


##MARS model

library(earth)
library(AppliedPredictiveModeling)
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:50)
set.seed(100)
marsTuned <- train(trainingData$x, trainingData$y,method = "earth", tuneGrid = marsGrid, trControl = trainControl(method = "cv"))
marsTuned
summary(marsTuned)
head(predict(marsTuned, testData$x))
plotmo(marsTuned)
plot(marsTuned)
varImp(marsTuned)



## 7.5 

##7.5

library(RANN)

data("ChemicalManufacturingProcess")


predictors<-subset(ChemicalManufacturingProcess, select = -Yield)
yield<-subset(ChemicalManufacturingProcess, select="Yield")

P1<- preProcess(predictors,method=c("knnImpute"))
Predictors1 <- predict(P1,predictors)

P2<- preProcess(Predictors1, method = c("center","scale"))
Predictors2 <- predict(P2,Predictors1)


Split<-createDataPartition(yield$Yield, p=0.8, list = FALSE)
TrainP<- Predictors2[Split,]
TrainY<-yield[Split,]

TestP<- Predictors2[-Split,]
TestY<-yield[-Split,]



##Neural Network with resampling

nnetGrid <- expand.grid(.decay = c(0, 0.01, .1),
                        .size = c(1:10),
                        ## The next option is to use bagging (see the
                        ## next chapter) instead of different random
                        ## seeds.
                        .bag = FALSE)
set.seed(100)
ctrl <- trainControl(method = "cv", number = 10)
nnetTune <- train(TrainP, TrainY,
                  method = "avNNet",
                  tuneGrid = nnetGrid,
                  trControl = ctrl,
                  ## Automatically standardize data prior to modeling
                  ## and prediction
                  preProc = c("center", "scale"),
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(TrainP) + 1) + 10 + 1,
                  maxit = 500)

nnetTune
plot(nnetTune)

Pred=predict(nnetTune, TestP)
SSEnnet=mean((TestY-Pred)^2)
RSSE=sqrt(SSEnnet)
Rsquared=(cor(TestY, Pred))^2
RSSE
Rsquared


##MARS with resampling

library(earth)
library(AppliedPredictiveModeling)

marsGrid2 <- expand.grid(.degree = 1:2, .nprune = 2:50)
set.seed(100)

marsTuned2 <- train(TrainP, TrainY,
                   method = "earth",
                   tuneGrid = marsGrid2,
                   trControl = trainControl(method = "cv"))

marsTuned2
summary(marsTuned2)
head(predict(marsTuned2, TestY))
##plotmo(marsTuned)
plot(marsTuned2)


varImp(marsTuned2)


Pred=predict(marsTuned2, TestP)
SSEmars=mean((TestY-Pred)^2)
RSSE=sqrt(SSEmars)
Rsquared=(cor(TestY, Pred))^2
RSSE
Rsquared


##SVM

install.packages("kernlab")
library(kernlab)
library(AppliedPredictiveModeling)

svmRTuned <- train(TrainP, TrainY,
                   method = "svmRadial",
                   preProc = c("center", "scale"),
                   tuneLength = 14,
                   trControl = trainControl(method = "cv"))


svmRTuned
plot(svmRTuned)

## The subobject named finalModel contains the model created by the ksvm
## function:

svmRTuned$finalModel

aa=varImp(svmRTuned)
plot(aa, top = 25, scales = list(y = list(cex = .95)))

Pred=predict(svmRTuned, TestP)
SSEsvm=mean((TestY-Pred)^2)
RSSE=sqrt(SSEsvm)
Rsquared=(cor(TestY, Pred))^2
RSSE
Rsquared



##knn

library(AppliedPredictiveModeling)
library(caret)
knnDescr <- TrainP[, -nearZeroVar(TrainP)]
set.seed(100)
knnTune <- train(knnDescr,
                 TrainY,
                 method = "knn",
                 # Center and scaling will occur for new predictions too
                 preProc = c("center", "scale"),
                 tuneGrid = data.frame(.k = 1:20),
                 trControl = trainControl(method = "cv"))
knnTune
plot(knnTune)

Pred=predict(knnTune, TestP)
SSEknn=mean((TestY-Pred)^2)
RSSE=sqrt(SSEknn)
Rsquared=(cor(TestY, Pred))^2
RSSE
Rsquared
