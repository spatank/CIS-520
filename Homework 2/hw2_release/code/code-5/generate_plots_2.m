% Submit your textual answers, and attach these plots in a latex file for
% this homework. 
% This script is merely for your convenience, to generate the plots for each
% experiment. Feel free to change it, as you do not need to submit this
% with your code.
clc; clear;
tic
% Loading the data: this loads X, and Ytrain.
load('X_noisy.mat'); % change this to X_noisy if you want to run the code on the noisy data
load('Y.mat');
X = X_noisy;
N_folds = 10;
K = [1, 3, 4, 6, 9, 14, 22, 35];
sigma = [1, 3, 5, 7, 9, 11];
errors_xval = zeros(100,size(K,2)); % errors_xval(i,j) records the [N_folds(j)]-fold cross validation error in trial i 
errors_test = zeros(100,size(K,2)); % errors_xval(i,j) records the true test error in trial i (the entire row will be identical)
errors_xval_kern = zeros(100,size(sigma,2)); % errors_xval(i,j) records the [N_folds(j)]-fold cross validation error in trial i 
errors_test_kern = zeros(100,size(sigma,2)); % errors_xval(i,j) records the true test error in trial i (the entire row will be identical)

distfunc = 'l2';
for trial = 1:100
        % split up the training data so that a quarter is held out to test
        % on
        train_indices = make_xval_partition(size(X,1), 4); % make four folds of data, 1 fold is held out to test on
        X_train = X(train_indices ~= 1, :); % train on all data not in fold 1
        Y_train = Y(train_indices ~= 1, :); 
        X_test = X(train_indices == 1, :); % test on all data in fold 1
        Y_true = Y(train_indices == 1, :);
        % the data that goes to fold 1 should be sufficiently randomized over the 100 trials  
    for i = 1:size(K,2)
            n_folds = N_folds;
            
            % make folds out of the training data
            part = make_xval_partition(size(X_train,1), n_folds);
        
            % run the cross-validation error function
            errors_xval(trial,i) = knn_xval_error(X_train, Y_train, K(i), part, distfunc);
        
            % run kNN/kernel-regression on all training data to compute test error
            [labels] = k_nearest_neighbours(X_train, Y_train, X_test, K(i), distfunc);
        
            errors_test(trial,i) = sum(labels ~= Y_true)/size(labels,1);
       % end
    end   
    
    for j = 1:size(sigma,2)
            n_folds = N_folds;
            
            % make folds out of the training data
            part = make_xval_partition(size(X_train,1), n_folds);
        
            % run the cross-validation error function
            errors_xval_kern(trial,j) = kernreg_xval_error(X_train, Y_train, sigma(j), part, distfunc);
        
            % run kNN/kernel-regression on all training data to compute test error
            [labels] = kernel_regression(X_train, Y_train, X_test, sigma(j));
        
            errors_test_kern(trial,j) = sum(labels ~= Y_true)/size(labels,1);
       % end
    end   
end

% K nearest neighbors
% code to plot the error bars. change these values depending on what
% experiment you are running
figure()
y = mean(errors_xval); e = std(errors_xval); x = K; % <- computes mean across all trials
errorbar(x, y, e);
hold on;
%figure()
y = mean(errors_test); e = std(errors_test); x = K; % <- computes mean across all trials
errorbar(x, y, e);
title('Original data, K = [1, 3, 4, 6, 9, 14, 22, 35];');
xlabel('K');
ylabel('Error');
legend('N-Fold Error','Test Error');
hold off;

% Kernel Regression
% code to plot the error bars. change these values depending on what
% experiment you are running
figure()
y = mean(errors_xval_kern); e = std(errors_xval_kern); x = sigma; % <- computes mean across all trials
errorbar(x, y, e);
hold on;
%figure()
y = mean(errors_test_kern); e = std(errors_test_kern); x = sigma; % <- computes mean across all trials
errorbar(x, y, e);
title('Original data, sigma = [1, 3, 5, 7, 9, 11];');
xlabel('Sigma');
ylabel('Error');
legend('N-fold Error','Test Error');
hold off;


toc