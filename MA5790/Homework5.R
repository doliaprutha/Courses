install.packages(c("glmnet", "pamr", "rms", "sparseLDA", "subselect", "caret", "MASS"))


#12.1

library(caret)
library(AppliedPredictiveModeling)
data(hepatic)

library(MASS)
library(e1071)
set.seed(1)

barplot(table(injury), main="Imbalanced Class Distribution")


#PreProcess the data
prebio <- preProcess(bio, method = c("center","scale","nzv","BoxCox"))
newbio <- predict(prebio,bio)

#Finding correlation
set.seed(1)
highCorBio<-findCorrelation(cor(newbio),cutoff = .80)
filteredCorBio <- newbio[,-highCorBio]

  
# Split the data

set.seed(1)

trainingRows =  createDataPartition(injury, p = .75, list= FALSE)

trainBio <- filteredCorBio[ trainingRows, ]
testBio <- filteredCorBio[-trainingRows, ]

trainInjury <- injury[trainingRows]
testInjury <- injury[-trainingRows]

#Model building

##Multinomial Logistic Regression##

set.seed(1)
ctrl <- trainControl(summaryFunction = defaultSummary)
lrBio <- train(x=trainBio,
               y = trainInjury,
               method = "multinom",
               metric = "Accuracy",
               trControl = ctrl)
summary(lrBio)

varImp(lrBio, scale = FALSE)

plot(lrBio)


predictionLRBio<-predict(lrBio,testBio)

confusionMatrix(data =predictionLRBio,
                reference = testInjury)


##Linear Discriminant Analysis

set.seed(1)

ldaBio <- train(x = trainBio,
                y = trainInjury,
                method = "lda",
                metric = "Accuracy",
                trControl = ctrl)

varImp(ldaBio, scale = FALSE)

summary(ldaBio)

##plot(ldaBio)

predictLDA <- predict(ldaBio,testBio)
confusionMatrix(data =predictLDA,
                reference = testInjury)


############## Partial Least Squares Discriminant Analysis ###############
library(MASS)
set.seed(1)
plsFit <- train(x = trainBio,
                y = trainInjury,
                method = "pls",
                tuneGrid = expand.grid(.ncomp = 1:1),
                preProc = c("center","scale"),
                metric = "Accuracy",
                trControl = ctrl)

plsFit

plot(plsFit)
summary(plsFit)

varImp(plsFit, scale = FALSE)

predictionPLSBio <-predict(plsFit,testBio)
confusionMatrix(data =predictionPLSBio,
                reference = testInjury)

############ Penalized Models ##########
## The primary package for penalized logistic regression is glmnet.

library(caret)


varImp(glmnTuned, scale = FALSE)

glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))

glmnTuned <- train(x=trainBio,
                        y =trainInjury,
                        method = "glmnet",
                        tuneGrid = glmnGrid,
                        preProc = c("center", "scale"),
                        metric = "Accuracy",
                        trControl = ctrl)

predictGlmnetBio <-  predict(glmnTuned,testBio)
confusionMatrix(data = predictGlmnetBio,
                reference = testInjury)

plot(glmnTuned, plotType = "level")

########### Penalized Models for LDA ###########

library(sparseLDA)
library(caret)
set.seed(1)

sparseLda <- sda(x=trainBio,y =trainInjury,lambda = 0.01,stop = -83)

## the ridge parameter called lambda.
## Lasso penalty is controled by stop. stop = -3 for three predictors.
## The argument method = "sparseLDA" can be used with train. In this case, train
## will tune the model over lambda and the number of retained predictors.

varImp(sparseLda, scale = FALSE)
summary(sparseLda)

predictSparseLDABio <-  predict(sparseLda,testBio)
confusionMatrix(data =predictSparseLDABio$class,
                reference = testInjury)



########### Nearest Shrunken Centroids ###########

library(pamr)
library(caret)

nscGrid <- data.frame(.threshold = seq(0,4, by=0.1))
set.seed(1)
nscTuned <- train(x = trainBio, y = trainInjury, method = "pam",
                  preProc = c("center", "scale"), tuneGrid = nscGrid,
                  metric = "Accuracy", trControl = ctrl)
nscTuned

plot(nscTuned)

summary(nscTuned)

predictNSC <-predict(nscTuned,testBio)
confusionMatrix(data =predictNSC,
                reference = testInjury)

predictors(nscTuned)

varImp(nscTuned, scale = FALSE)



#12.3
library(C50)
library(corrplot)
data(churn)
str(churnTrain)
table(churnTrain$churn)
#plot
par(mfrow = c(2,2))
plot(churn~state,data = churnTrain)
plot(churn~account_length,data = churnTrain)
plot(churn~area_code,data = churnTrain)
plot(churn~international_plan,data = churnTrain)
par(mfrow = c(2,2))
plot(churn~voice_mail_plan,data = churnTrain)
plot(churn~number_vmail_messages,data = churnTrain)
plot(churn~total_day_minutes,data = churnTrain)
plot(churn~total_day_calls,data = churnTrain)
par(mfrow = c(2,2))
plot(churn~total_day_charge,data = churnTrain)
plot(churn~total_eve_minutes,data = churnTrain)
plot(churn~total_eve_calls,data = churnTrain)
plot(churn~total_eve_charge,data = churnTrain)
par(mfrow = c(2,2))
plot(churn~total_night_minutes,data = churnTrain)
plot(churn~total_night_calls,data = churnTrain)
plot(churn~total_night_charge,data = churnTrain)
plot(churn~total_intl_minutes,data = churnTrain)
par(mfrow = c(1,3))
plot(churn~total_intl_calls,data = churnTrain)
plot(churn~total_intl_charge,data = churnTrain)
plot(churn~number_customer_service_calls,data = churnTrain)
#(c)
predict_train <- churnTrain[,-20]
ctrain <- churnTrain[,20]
predict_test <- churnTest[,-20]
ctest <- churnTest[,20]
#Dummy Variables
library(caret)
dummy <- dummyVars("~state + area_code + international_plan + voice_mail_plan",
                   data = predict_train, fullRank = TRUE)
dummytrain <- data.frame(predict(dummy, newdata = predict_train))
dummy <- dummyVars("~state + area_code + international_plan + voice_mail_plan",
                   data = predict_test, fullRank = TRUE)
dummytest <- data.frame(predict(dummy, newdata = predict_test))
# Drop all factor predictors:
predict_train <- predict_train[,-c(1,3,4,5)]
predict_test <- predict_test[,-c(1,3,4,5)]
predict_train <- merge(predict_train, dummytrain, by =0)
predict_test <- merge(predict_test, dummytest, by =0)
predict_train <- predict_train[,-c(1)]
predict_test <- predict_test[,-c(1)]
#PreProcess
newdata <- preProcess(predict_train, method = c("center","scale","BoxCox"))
newdata_train <- predict(newdata, predict_train)
newdata_test <- predict(newdata, predict_test)
NV3 <- nearZeroVar(newdata_train)
newdata_train <- newdata_train[-NV3]
newdata_test <- newdata_test[-NV3]
highcor <- cor(newdata_train)
highcorpredict <- findCorrelation(highcor)
newdata_train <- newdata_train[,-highcorpredict]
newdata_test <- newdata_test[,-highcorpredict]
library(pROC)
ctrl1 = trainControl(method = "LGOCV",
                     summaryFunction=twoClassSummary,
                     classProbs=TRUE )
# Logistic Regression Model:
set.seed(1)
lrnew <- train(x=newdata_train,
               y = ctrain,
               method = "glm",
               preProc = c("center", "scale"),
               metric = "ROC",
               trControl = ctrl1)
summary(lrnew)
lrnew
predictionLR<-predict(lrnew,newdata_test)
confusionMatrix(data =predictionLR,
                reference = ctest)
# Linear Discriminant Analysis:
set.seed(1)
lda <- train(x = newdata_train,
             y = ctrain,
             method = "lda",
             preProc = c("center", "scale"),
             metric = "ROC",
             trControl = ctrl1)
lda
summary(lda)
predictionLDA <- predict(lda,newdata_test)
confusionMatrix(data =predictionLDA,
                reference = ctest)
############## Partial Least Squares Discriminant Analysis ###############
library(MASS)
set.seed(1)
plsFitc <- train(x = newdata_train,
                 y = ctrain,
                 method = "pls",
                 tuneGrid = expand.grid(.ncomp = 1:1),
                 preProc = c("center","scale"),
                 metric = "ROC",
                 trControl = ctrl1)
plsFitc
summary(plsFitc)
varImp(plsFitc, scale = FALSE)
predictionPLS <-predict(plsFitc,newdata_test)
confusionMatrix(data =predictionPLS,
                reference = ctest)
# Penalized Methods:
library(caret)
set.seed(1)
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))
glmnTuned <- train(x = newdata_train, y = ctrain, method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = c("center", "scale"),
                   metric = "ROC", trControl = ctrl1)
glmnTuned
important <- varImp(glmnTuned, scale = FALSE)
plot(important, top = 5, scales = list(y = list(cex = .95)))
predictGlmnet <- predict(glmnTuned,newdata_test)
confusionMatrix(data = predictGlmnet,
                reference = ctest)
plot(glmnTuned, plotType = "level")
# Nearest shrunken Centroids:
library(pamr)
nscGrid <- data.frame(.threshold = seq(0,4, by=0.1))
set.seed(1)
nscTunedc <- train(x = newdata_train, y = ctrain, method = "pam",
                   preProc = c("center", "scale"), tuneGrid = nscGrid,
                   metric = "ROC", trControl = ctrl1)
nscTunedc
plot(nscTunedc)
summary(nscTunedc)
predictNSCc <-predict(nscTunedc,newdata_test)
confusionMatrix(data =predictNSCc,
                reference = ctest)
predictors(nscTunedc)


