# AppliedPredictiveModeling, March 2019
# https://github.com/cran/AppliedPredictiveModeling/blob/master/inst/chapters/04_Over_Fitting.Rout
#
# R Bloggers Article
# https://www.r-bloggers.com/classification-on-the-german-credit-database/
#

# install.packages("AppliedPredictiveModeling")
# install.packages("caret")
# install.packages("kernlab")
# install.packages("e1071")
# install.packages("doMC", repos="http://R-Forge.R-project.org")

library(caret)
data(GermanCredit)
set.seed(1056)

GermanCredit <- GermanCredit[, -nearZeroVar(GermanCredit)]
GermanCredit$CheckingAccountStatus.lt.0 <- NULL
GermanCredit$SavingsAccountBonds.lt.100 <- NULL
GermanCredit$EmploymentDuration.lt.1 <- NULL
GermanCredit$EmploymentDuration.Unemployed <- NULL
GermanCredit$Personal.Male.Married.Widowed <- NULL
GermanCredit$Property.Unknown <- NULL
GermanCredit$Housing.ForFree <- NULL

set.seed(100)
inTrain <- createDataPartition(GermanCredit$Class, p = .8)[[1]]
GermanCreditTrain <- GermanCredit[ inTrain, ]
GermanCreditTest  <- GermanCredit[-inTrain, ]

library(kernlab)
set.seed(231)
sigDist <- sigest(Class ~ ., data = GermanCreditTrain, frac = 1)
svmTuneGrid <- data.frame(sigma = as.vector(sigDist)[1], C = 2^(-2:7))

library(doMC)
registerDoMC(4)
 
set.seed(1056)
svmFit <- train(Class ~ .,
                 data = GermanCreditTrain,
                 method = "svmRadial",
                 preProc = c("center", "scale"),
                 tuneGrid = svmTuneGrid,
                 trControl = trainControl(method = "repeatedcv", 
                 repeats = 5,
                 classProbs = TRUE))

svmFit
 
plot(svmFit, scales = list(x = list(log = 2)))

predictedClasses <- predict(svmFit, GermanCreditTest)
str(predictedClasses)

predictedProbs <- predict(svmFit, newdata = GermanCreditTest, type = "prob")
head(predictedProbs)

# set.seed(1056)
# svmFit10CV <- train(Class ~ .,
#                    data = GermanCreditTrain,
#                    method = "svmRadial",
#                    preProc = c("center", "scale"),
#                    tuneGrid = svmTuneGrid,
#                    trControl = trainControl(method = "cv", 
#                    number = 10))
#
# svmFit10CV
#
# set.seed(1056)
# svmFitLOO <- train(Class ~ .,
#                  data = GermanCreditTrain,
#                   method = "svmRadial",
#                   preProc = c("center", "scale"),
#                   tuneGrid = svmTuneGrid,
#                   trControl = trainControl(method = "LOOCV"))
# svmFitLOO
#
# set.seed(1056)
# svmFitLGO <- train(Class ~ .,
#                   data = GermanCreditTrain,
#                   method = "svmRadial",
#                   preProc = c("center", "scale"),
#                   tuneGrid = svmTuneGrid,
#                   trControl = trainControl(method = "LGOCV", 
#                   number = 50, 
#                   p = .8))
# svmFitLGO
# 
# set.seed(1056)
# svmFitBoot <- train(Class ~ .,
#                     data = GermanCreditTrain,
#                     method = "svmRadial",
#                     preProc = c("center", "scale"),
#                     tuneGrid = svmTuneGrid,
#                     trControl = trainControl(method = "boot", number = 50))
# svmFitBoot
#
# set.seed(1056)
# svmFitBoot632 <- train(Class ~ .,
#                       data = GermanCreditTrain,
#                       method = "svmRadial",
#                       preProc = c("center", "scale"),
#                       tuneGrid = svmTuneGrid,
#                       trControl = trainControl(method = "boot632", 
#                       number = 50))
# svmFitBoot632

set.seed(1056)
glmProfile <- train(Class ~ .,
                    data = GermanCreditTrain,
                    method = "glm",
                    trControl = trainControl(method = "repeatedcv", 
                    repeats = 5))

glmProfile

resamp <- resamples(list(SVM = svmFit, Logistic = glmProfile))
summary(resamp)
modelDifferences <- diff(resamp)
summary(modelDifferences)

modelDifferences$statistics$Accuracy$SVM.diff.Logistic

# sessionInfo()
# q("no")

proc.time()
