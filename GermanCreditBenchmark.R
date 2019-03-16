# https://www.r-bloggers.com/classification-on-the-german-credit-database/

url="http://freakonometrics.free.fr/german_credit.csv"
credit=read.csv(url, header = TRUE, sep = ",")

str(credit)

# Convert categorical variables as factors
F=c(1,2,4,5,7,8,9,10,11,12,13,15,16,17,18,19,20)
for(i in F) credit[,i]=as.factor(credit[,i])

i_test=sample(1:nrow(credit),size=333)
i_calibration=(1:nrow(credit))[-i_test]

# The first model we can fit is a logistic regression, on selected covariates

LogisticModel <- glm(Creditability ~ Account.Balance + 
                     Payment.Status.of.Previous.Credit + Purpose + 
                     Length.of.current.employment + 
                     Sex...Marital.Status, 
                     family=binomial, 
                     data = credit[i_calibration,])

fitLog <- predict(LogisticModel,type="response", newdata=credit[i_test,])

library(ROCR)
pred = prediction( fitLog, credit$Creditability[i_test])
perf <- performance(pred, "tpr", "fpr")
plot(perf)
AUCLog1=performance(pred, measure = "auc")@y.values[[1]]
cat("AUC: ",AUCLog1,"n")

# An alternative is to consider a logistic regression on all explanatory variables

LogisticModel <- glm(Creditability ~ ., family=binomial, data = credit[i_calibration,])

# We might overfit, here, and we should observe that on the ROC curve

fitLog <- predict(LogisticModel, type="response", newdata=credit[i_test,])

pred = prediction( fitLog, credit$Creditability[i_test])
perf <- performance(pred, "tpr", "fpr")
plot(perf)
AUCLog2=performance(pred, measure = "auc")@y.values[[1]]
cat("AUC: ",AUCLog2,"n")

# Consider now some regression tree (on all covariates)

library(rpart)
ArbreModel <- rpart(Creditability ~ ., data = credit[i_calibration,])

# We can visualize the tree using

library(rpart.plot)
prp(ArbreModel,type=2,extra=1)

# The ROC curve for that model is

fitArbre <- predict(ArbreModel, newdata=credit[i_test,], type="prob")[,2]
pred = prediction( fitArbre, credit$Creditability[i_test])
perf <- performance(pred, "tpr", "fpr")
plot(perf)
AUCArbre=performance(pred, measure = "auc")@y.values[[1]]
cat("AUC: ",AUCArbre,"n")

library(randomForest)
RF <- randomForest(Creditability ~ ., data = credit[i_calibration,])
fitForet <- predict(RF, newdata=credit[i_test,], type="prob")[,2]
pred = prediction( fitForet, credit$Creditability[i_test])
perf <- performance(pred, "tpr", "fpr")
plot(perf)
AUCRF=performance(pred, measure = "auc")@y.values[[1]]
cat("AUC: ",AUCRF,"n")


AUC = function(i){
      set.seed(i)
      i_test=sample(1:nrow(credit),size=333)
      i_calibration=(1:nrow(credit))[-i_test]
      LogisticModel <- glm(Creditability ~ ., family=binomial, data = credit[i_calibration,])
      # summary(LogisticModel)
      fitLog <- predict(LogisticModel,type="response", newdata=credit[i_test,])
      # library(ROCR)
      pred = prediction(fitLog, credit$Creditability[i_test])
      AUCLog2=performance(pred, measure = "auc")@y.values[[1]] 
      RF <- randomForest(Creditability ~ ., data = credit[i_calibration,])
      fitForet <- predict(RF, newdata=credit[i_test,], type="prob")[,2]
      pred = prediction( fitForet, credit$Creditability[i_test])
      AUCRF=performance(pred, measure = "auc")@y.values[[1]]
      return(c(AUCLog2,AUCRF))
}

A=Vectorize(AUC)(1:20)
plot(t(A))
