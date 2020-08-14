%% Method 2: Random

clc; close all; clear;
load('data_new1.mat')

indic_random = ismissing(Xtrain_random);
Xtrain_random_indic = reshape([Xtrain_random;indic_random], size(Xtrain_random,1), []);

tree_random_indic = fitctree(Xtrain_random_indic, Ytrain);

% training accuracy
y_hat_train_random_indic = predict(tree_random_indic,Xtrain_random_indic);
train_accuracy_random_indic = sum(y_hat_train_random_indic == Ytrain) / size(Ytrain,1);

indic_random_test = ismissing(Xtest_random);
Xtest_random_indic = reshape([Xtest_random;indic_random_test], size(Xtest_random,1), []);

% testing accuracy
y_hat_test_random_indic = predict(tree_random_indic,Xtest_random_indic);
test_accuracy_random_indic = sum(y_hat_test_random_indic == Ytest) / size(Ytest,1);

%% Method 2: Not-Random

indic_nrandom = ismissing(Xtrain_nrandom);
Xtrain_nrandom_indic = reshape([Xtrain_nrandom;indic_nrandom], size(Xtrain_nrandom,1), []);

tree_nrandom_indic = fitctree(Xtrain_nrandom_indic, Ytrain);

% training accuracy
y_hat_train_nrandom_indic = predict(tree_nrandom_indic,Xtrain_nrandom_indic);
train_accuracy_nrandom_indic = sum(y_hat_train_nrandom_indic == Ytrain) / size(Ytrain,1);

indic_nrandom_test = ismissing(Xtest_nrandom);
Xtest_nrandom_indic = reshape([Xtest_nrandom;indic_nrandom_test], size(Xtest_nrandom,1), []);

% testing accuracy
y_hat_test_nrandom_indic = predict(tree_nrandom_indic,Xtest_nrandom_indic);
test_accuracy_nrandom_indic = sum(y_hat_test_nrandom_indic == Ytest) / size(Ytest,1);
