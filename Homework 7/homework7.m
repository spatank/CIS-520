clc; close all; clear;
load('data_new1.mat')

%% Full Tree Accuracy

tree_full = fitctree(Xtrain_full, Ytrain);

% training accuracy
y_hat_train_full = predict(tree_full,Xtrain_full);
train_accuracy_full = sum(y_hat_train_full == Ytrain) / size(Ytrain,1);

% testing accuracy
y_hat_test_full = predict(tree_full,Xtest_full);
test_accuracy_full = sum(y_hat_test_full == Ytest) / size(Ytest,1);

%%  Method 1: Random

mean_Xtrain_random = nanmean(Xtrain_random,1);

for i = 1:size(mean_Xtrain_random,2)
    col = Xtrain_random(:,i);
    idx = isnan(col);
    col(idx) = mean_Xtrain_random(i);
    Xtrain_random(:,i) = col;
end
tree_random = fitctree(Xtrain_random, Ytrain);

% training accuracy
y_hat_train_random = predict(tree_random,Xtrain_random);
train_accuracy_random = sum(y_hat_train_random == Ytrain) / size(Ytrain,1);

% testing accuracy

for i = 1:size(mean_Xtrain_random,2)
    col = Xtest_random(:,i);
    idx = isnan(col);
    col(idx) = mean_Xtrain_random(i);
    Xtest_random(:,i) = col;
end

y_hat_test_random = predict(tree_random,Xtest_random);
test_accuracy_random = sum(y_hat_test_random == Ytest) / size(Ytest,1);

%%  Method 1: Not-Random

mean_Xtrain_nrandom = nanmean(Xtrain_nrandom,1);

for i = 1:size(mean_Xtrain_nrandom,2)
    col = Xtrain_nrandom(:,i);
    idx = isnan(col);
    col(idx) = mean_Xtrain_nrandom(i);
    Xtrain_nrandom(:,i) = col;
end
tree_nrandom = fitctree(Xtrain_nrandom, Ytrain);

% training accuracy
y_hat_train_nrandom = predict(tree_nrandom,Xtrain_nrandom);
train_accuracy_nrandom = sum(y_hat_train_nrandom == Ytrain) / size(Ytrain,1);

% testing accuracy

for i = 1:size(mean_Xtrain_nrandom,2)
    col = Xtest_nrandom(:,i);
    idx = isnan(col);
    col(idx) = mean_Xtrain_nrandom(i);
    Xtest_nrandom(:,i) = col;
end

y_hat_test_nrandom = predict(tree_nrandom,Xtest_nrandom);
test_accuracy_nrandom = sum(y_hat_test_nrandom == Ytest) / size(Ytest,1);




