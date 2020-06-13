# Install packages if required
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
# Load libraries
library(tidyverse)
library(caret)
library(corrplot)
library(stringr)
library(rpart)
library(gbm)

# Load data file from github
dl <- tempfile()
download.file("https://github.com/limhongwei/lol/raw/master/games.csv", dl)
games <- read.csv(dl, stringsAsFactors = FALSE)

# Brief look at dataset
glimpse(games)

# Check NA values
sum(sapply(1:ncol(games), function(x) sum(is.na(games[,x]))))

# Tidy data by adding redWins column, gathering the variables, 
temp <- games %>% 
  mutate(redWins = ifelse(blueWins==1,0,1)) %>% 
  gather(key, value, -gameId)
# Add in "_" separator for 'blue' and 'red' teams for easier separation
temp$key <- str_replace_all(temp$key, "blue", "blue_")
temp$key <- str_replace_all(temp$key, "red", "red_")
# Separate by team
games_clean <- temp %>% 
  separate(key, c("team", "variable"), "_") %>%
  spread(variable, value)

# Brief look at tidy data
glimpse(games_clean)

# Win rate of choosing Blue team should be 0.5
mean(games$blueWins==1)

# Check derived variables GoldPerMin, CSPerMin and EliteMonsters
games_clean %>% 
  mutate(count1 = GoldPerMin - TotalGold/10,
         count2 = CSPerMin - TotalMinionsKilled/10,
         count3 = EliteMonsters - Dragons - Heralds) %>% 
# All summations should be zero if the variables are derived from other variables  
  summarize(count1 = sum(count1), 
            count2 = sum(count2), 
            count3 = sum(count3))

# Check near zero variance variables
nearZeroVar(games_clean, saveMetrics = TRUE)

# Remove irrelevant, derived and near zero variance variables
games_clean <- games_clean %>% 
  select(-gameId, -team, -GoldPerMin, -CSPerMin, -EliteMonsters, -TowersDestroyed) %>%
  mutate(Wins = factor(Wins))

# Brief look at data ready for exploration
str(games_clean)

# First split of data into 80% main set and 20% validation set
# The main set will be further split into 90/10 before model training below
set.seed(2020, sample.kind = "Rounding") 
# if using R 3.5 or earlier, use `set.seed(2020)` instead
test_index <- createDataPartition(games_clean$Wins, times = 1, p = 0.2, list = FALSE)
main <- games_clean[-test_index,]
validation <- games_clean[test_index,]

# Win rate of choosing either Blue or Red should be 0.5
mean(main$Wins==1)

# Correlation among predictors
temp <- main %>% select(-Wins)
tempcor <- cor(temp)
corrplot::corrplot(tempcor, type = "lower")

# Distribution of predictors
main %>% gather(key, value, -Wins) %>% 
  ggplot(aes(value, fill = Wins)) + 
  geom_density(aes(y = ..density..), kernel = "gaussian", alpha = 0.6) +
  facet_wrap(~key, scales = "free")

# Range of 'WardsPlaced' and 'WardsDestroyed' seem to be far-fetched
range(main$WardsPlaced)
range(main$WardsDestroyed)

# # Variable importance using Random Forest
# The plot is shown in PDF and RMD, hidden in R code because of long computation time
# main_rf <- train(Wins ~ ., method = "rf",
#                  data = main,
#                  ntree = 100)
# plot(varImp(main_rf), main = "Variable Importance using Random Forest")
# 
# # Variable importance using Logistic Regression
# The plot is shown in PDF and RMD, hidden in R code because of possible long computation time
# main_glm <- train(Wins ~ ., 
#                   method = "glm", 
#                   data = main)
# plot(varImp(main_glm), main = "Variable Importance using Logistic Regression")
  
# Second split of 90/10 for model training (90% train_set and 10% test_set)
set.seed(2020, sample.kind = "Rounding") 
# if using R 3.5 or earlier, use `set.seed(2020)` instead
test_index <- createDataPartition(main$Wins, times = 1, p = 0.1, list = FALSE)
train_set <- main[-test_index,]
test_set <- main[test_index,]

# Model training of train_set vs test_set
# A variety of models suitable for classification tasks is chosen
# This code could take a while to load
models <- c("naive_bayes", "gamLoess", "lda", "qda", "glm", "glmboost",
            "pls", "multinom", "knn", "gbm", "rpart")
fits <- lapply(models, function(model){
  set.seed(2020, sample.kind = "Rounding") 
  # if using R 3.5 or earlier, use `set.seed(2020)` instead
  print(model)
  train(Wins ~ ., method = model, data = train_set)
})
names(fits) <- models
# Predicting win rates on test_set
pred <- sapply(fits, function(object) predict(object, newdata = test_set))

# Generate output of model training
# Extracting Accuracy, Sensitivity and Specificity from confusionMatrix of all models
out <- sapply(seq(1,length(models),1), function(x){
  cm <- confusionMatrix(factor(pred[,x]), test_set$Wins)
  acc <- cm$overall["Accuracy"]
  sens <- cm$byClass["Sensitivity"]
  spec <- cm$byClass["Specificity"]
  return(c(acc, sens, spec))
})

# Generate results of model training
# Collating the results into a tibble
labels <- sapply(seq(1,length(models),1), function(x) getModelInfo(models[x])[[1]]$label)
results <- tibble(Model = models, Label = labels)
temp <- out %>% t %>% as.data.frame()
results <- cbind(results, temp)

# Comparing models in terms of Accuracy and Kappa
models_compare <- resamples(fits)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)

# Ensemble by Average (using mean accuracy from all models)
acc_avg <- mean(results$Accuracy)
acc_avg

# Ensemble by Majority Vote
# Create a vote - if more than half of the models predict 0, then y_hat (predicted win rate) = 0;
# if half or less of the models predict 1, then y_hat = 1
votes <- rowMeans(pred=="0")
y_hat <- ifelse(votes > 0.5, "0", "1")
acc_mv <- mean(y_hat == test_set$Wins)
acc_mv

# Remove models with more than 0.75 correlation
# using modelCor function to check cross-correlation among all the models
a <- modelCor(models_compare)
# using findCorrelation function to find correlation pair of 0.75 or above
x <- findCorrelation(a, cutoff = .75)
# remove models with high correlation
selected <- colnames(a[,-x])
# Ensemble by Stacking
# Fit selected models into train_set (instead of test_set)
pred_train <- sapply(fits[selected], function(object) predict(object, newdata = train_set))
# Add predicted values into train_set
stack_train <- train_set %>% cbind(pred_train)
# Add predicted values (from initial model training) into test_set
stack_test <- test_set %>% cbind(pred[,selected])
set.seed(2020, sample.kind = "Rounding") 
# if using R 3.5 or earlier, use `set.seed(2020)` instead
# Fit glm (top layer) into predicted values (bottom layer of all models)
model_glm <- train(stack_train[selected], stack_train$Wins, method = "glm")
y_hat_stack <- predict(model_glm, stack_test[selected])
acc_stack <- mean(y_hat_stack==stack_test$Wins)
acc_stack

# Result of different ensembles
ensem <- tibble(Method = c("Ensemble by Average",
                           "Ensemble by Majority Vote",
                           "Ensemble by Stacking"),
                Accuracy = c(acc_avg, acc_mv, acc_stack))

# Test final model using validation set
fits <- lapply(models, function(model){
  set.seed(2020, sample.kind = "Rounding") 
  # if using R 3.5 or earlier, use `set.seed(2020)` instead
  print(model)
  train(Wins ~ ., method = model, data = validation)
})
names(fits) <- models
pred <- sapply(fits, function(object) predict(object, newdata = validation))

# Generate result of model training on validation set
# for reference only
# out <- sapply(seq(1,length(models),1), function(x){
#   cm <- confusionMatrix(factor(pred[,x]), validation$Wins)
#   acc <- cm$overall["Accuracy"]
#   sens <- cm$byClass["Sensitivity"]
#   spec <- cm$byClass["Specificity"]
#   return(c(acc, sens, spec))
# })

# Results of different models on validation set
# for reference only
# labels <- sapply(seq(1,length(models),1), function(x) getModelInfo(models[x])[[1]]$label)
# results <- tibble(Model = models, Label = labels)
# temp <- out %>% t %>% as.data.frame()
# results <- cbind(results, temp)

# Ensemble with majority vote on validation set
votes <- rowMeans(pred=="0")
y_hat <- ifelse(votes > 0.5, "0", "1")
acc_val <- mean(y_hat == validation$Wins)
acc_val
