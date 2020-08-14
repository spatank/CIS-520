% Submit your textual answers, and attach these plots in a latex file for
% this homework. 
% This script is merely for your convenience, to generate the plots for each
% experiment. Feel free to change it, as you do not need to submit this
% with your code.
clc; close all; clear;
tic
% Loading the data: this loads X, and Ytrain.
load('../data/X.mat'); % change this to X_noisy if you want to run the code on the noisy data
load('../data/Y.mat');

N_folds = [3, 5, 10, 15];
errors_xval = zeros(100,size(N_folds,2)); % errors_xval(i,j) records the [N_folds(j)]-fold cross validation error in trial i 
errors_test = zeros(100,size(N_folds,2)); % errors_xval(i,j) records the true test error in trial i (the entire row will be identical)
K = 1;
sigma = 1;
distfunc = 'l2';
for trial = 1:100
        % split up the training data so that a quarter is held out to test
        train_indices = make_xval_partition(size(X,1), 4); % make four folds of data, 1 fold is held out to test on
        X_test = X(train_indices == 1, :); % test on all data in fold 1
        X_train = X(train_indices ~= 1, :); % train on all data not in fold 1
        Y_train = Y(train_indices ~= 1, :); 
        Y_true = Y(train_indices == 1, :);
        % the data that goes to fold 1 should be sufficiently randomized over the 100 trials  
    for i = 1:numel(N_folds)
        n_folds = N_folds(i);
        % make folds out of the training data
        part = make_xval_partition(size(X_train,1), n_folds);
        
        % run the cross-validation error function
        errors_xval(trial,i) = knn_xval_error(X_train, Y_train, K, part, distfunc);
        % errors_xval(trial,i) = kernreg_xval_error(X_train, Y_train, sigma, part, distfunc);
        
        % run kNN/kernel-regression on all training data to compute test error
        [labels] = k_nearest_neighbours(X_train, Y_train, X_test, K, distfunc);
        % [labels] = kernel_regression(X_train, Y_train, X_test, sigma);
        
        errors_test(trial,i) = sum(labels ~= Y_true)/size(labels,1);
    end   
end

% code to plot the error bars. change these values depending on what
% experiment you are running
y = mean(errors_xval); e = std(errors_xval); x = [3, 5, 10, 15]; % <- computes mean across all trials
errorbar(x, y, e);
hold on;
y = mean(errors_test); e = std(errors_test); x = [3, 5, 10, 15]; % <- computes mean across all trials
errorbar(x, y, e);
title('Original data, N = [3, 5, 10, 15]');
xlabel('N');
ylabel('Error');
legend('N-Fold Error','Test Error');
hold off;
toc