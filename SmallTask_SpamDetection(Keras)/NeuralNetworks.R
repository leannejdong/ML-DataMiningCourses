# Notes for nnet
# M is the number of hidden units. By making M large enough, neural networks 
# allow one to fit almost arbitrarily flexible. another example of a neural network 
# fit where the test error rate starts to increase as we increase the number of hidden units.
# This penalty (called weight decay) forces the neural network fit to be smoother. 
# The penalty function: once we use weight decay in the fit the error rate becomes 
# fairly insensitive to the number of hidden units (as long as we have enough).

# use {nnet} to Neural Networks
library(ElemStatLearn)
data(spam)
# alternatively read the data
# spam = read.table("spam.data", header=T)
dim(spam)
summary(spam)
set.seed(42) 
my.sample <- sample(nrow(spam), 3221) 
spam.train <- spam[my.sample, ] 
spam.test <- spam[-my.sample, ]
write.csv(spam.train,'spam_train.csv')
write.csv(spam.test,'spam_test.csv')
tr = sample(1:150,100)
library(nnet)
# The nnet() function implements a single hidden layer neural network. The
# syntax is almost identical to that for lm() etc.
# There are five possible additional variables to feed nnet(). First we must tell
# it how many hidden units to use through the command size=?. We can also specify a
# decay value by decay=? (by default this is zero if we don't specify anything).
# By default the maximum number of iterations it
# does is 100. You can change the default using maxit=?. Finally, if you
# are using nnet for a regression (rather than a classification) problem
# you need to set linout=T to tell nnet to use a linear output (rather
# than a sigmoid output that is used for a classification situation).

# Fit a neural network to your training data using 2 hidden units and 0 decay.
# decay parameter for weight decay. Default 0. It is the penalty on large coefficient
nn1 <- nnet(formula = spam ~ ., data=spam.train, size=2, decay=0.1, maxit=1000) 

summary(nn1)
# Refit the neural network with 10 hidden units and 0 decay.
nnfit = nnet(spam ~ ., spam, size = 10, subset = tr);

# predict spam.test dataset on nn1 
nn1.pr.test <- predict(nn1, spam.test, type='class') 

nn1.test.tab<-table(spam.test$spam, nn1.pr.test, dnn=c('Actual', 'Predicted')) 
nn1.test.tab
# Calucate overall error percentage ~ 7.68% 
nn1.test.perf <- 100 * (nn1.test.tab[2] + nn1.test.tab[3]) / sum(nn1.test.tab)
nn1.test.perf


