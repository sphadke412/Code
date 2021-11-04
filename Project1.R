pacman::p_load('ISLR', 'tidyverse', 'ggthemes', 'caret', 'doMC', 
               'gbm', 'knitr', 'MASS')

packages <- c("dplyr", "ggplot2", "readr")
sapply(packages, require, character.only = TRUE)

#Q1:
#Q2:
```{r}
data("Hitters")


Hitters %>%
  filter(!is.na(Salary)) %>%
  mutate(logSalary = log(Salary)) %>%
  dplyr::select(-Salary) -> Hitters

```
#Q1: 272 observations were omitted from the data set
#Q2: The log transformation is used to get rid of any outliers in a highly
#skewed distribution.  It allows the data to be more normally distributed. 

#Q3:
```{r}
input <- Hitters[,c('Years', 'Hits', 'logSalary')]
print(head(input))

library(ggplot2)
ggplot() + geom_point(data = Hitters, aes(x = Years, y = Hits, 
                                 color = logSalary)) + ggtitle("Years vs Hits")

```


#Q4:
```{r}

lm(Hitters$logSalary~., data = Hitters)


library(leaps)
reg_Salary <- regsubsets(Hitters$logSalary~., data = Hitters)
summary(reg_Salary)
sum <- summary(reg_Salary)
sum$bic
```
#The best model according to  BIC  would consist of two variables based on the 
#lowest BIC values which include variables HITS and CATBats. 


#Q5
```{r}
library(caret)
set.seed(42)
trainindex <- createDataPartition(Hitters$logSalary, p = 0.8, list = FALSE)
hitters.train <- Hitters[trainindex, ]
hitters.valid <- Hitters[-trainindex, ]

```
#Q6
```{r}
library(rpart)

tree <- rpart(logSalary~ Years+Hits, data = hitters.train, method = "anova")
summary(tree)

plot(tree, uniform = TRUE, main = "Regression Tree")  
text(tree, use.n = TRUE, all= TRUE, cex = .8)

pred.tree<- predict(tree, hitters.valid)
mean((hitters.valid$logSalary - pred.tree)^2)

players.highest <- pred.tree[pred.tree == max(pred.tree)]
names(players.highest)
```
#The last player Willie Wilson will have the highest salary because the log 
#value of the salary is 6.90 which will translate to the highest salary. The 
#top three are Willie Wilson, Willie Upshaw and Wayne Tolleson. Rule:
#years >= 4.5 AND hits >= 117.5 AND Years >= 6.5 and Hits >= 50.5. Players with
#more experience and more hits will earn a higher salary. 






#Q7
```{r}
library(rpart)
library(tree)
set.seed(42)

hitters.tree <- tree(logSalary~., data = Hitters, method = "anova")
summary(hitters.tree)

cv.Hitters <- cv.tree(hitters.tree)
prune.Hitters <- prune.tree(hitters.tree, best =7)
yhat <- predict(hitters.tree, data = hitters.train)




library(gbm)
set.seed(42)
tree.hitters <- tree(logSalary ~., Hitters, subset = trainindex)
summary(tree.hitters)

plot(tree.hitters) 
text(tree.hitters, pretty = 0)

shrinkparameter <- seq(-10, -0.2, by = 0.1)
lambdas <- 10^ shrinkparameter
length.lambdas <- length(lambdas)
train.error <- rep(NA, length.lambdas)
test.error <- rep(NA, length.lambdas)

for (i in 1:length.lambdas) {
  boost_hitters <- gbm(logSalary~., data = hitters.train, 
                       distribution = "gaussian", n.trees = 1000, 
                       shrinkage = lambdas[i])
  train.predict <- predict(boost_hitters, hitters.train, n.trees = 1000)
  test.predict <- predict(boost_hitters, hitters.valid, n.trees = 1000)
  train.error[i] <- mean((hitters.train$logSalary - train.predict)^2)
  test.error[i] <- mean((hitters.valid$logSalary - test.predict)^2)
  
}


plot(lambdas, train.error, type = "b", xlab = "Shrinkage", ylab = "Train MSE",
     col = "blue", pch = 20, bty = "n") 
     
plot(lambdas, test.error, type = "b", xlab = "Shrinkage", ylab = "Test MSE",
     col = "blue", pch = 20, bty = "n") 

min(test.error)
lambdas[which.min(test.error)]





```

#Q9

```{r}
library(gbm)
set.seed(42)

boost.hitters <- gbm(logSalary~., hitters.train, 
                     distribution = "gaussian", n.trees = 1000, 
                     interaction.depth = 4)


summary(boost.hitters)
```
#According to the result, CHits appears to be the most important predictor in 
# the boosted model with a value of 20.144.




#Q10

```{r}
library(randomForest)
set.seed(42)

bag.hitters <- randomForest(logSalary~., hitters.train, mtry = 19, 
                            importance= TRUE, na.action = na.roughfix)
bag.hitters

yhat.bag <- predict(bag.hitters, hitters.valid)
mean((yhat.bag - hitters.valid$logSalary)^2)
```
#The test for MSE with bagging is 0.13 which is slightly lower then the test 
#test for MSE for boosting which was 0.159. 






















