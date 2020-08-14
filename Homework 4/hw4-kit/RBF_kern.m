clc; close all; clear;

%% Pick a C for each sigma using CV
sigma = [0.1, 1, 10, 100, 1000];
C = [1, 10, 10^2, 10^3, 10^4, 10^5];
error = zeros(size(sigma, 2), 5, size(C, 2));
cd Synthetic/CrossValidation
% make sure to be in the CrossValidation folder for the loop to work
for i = 1:size(sigma, 2)
    sig = sigma(i);
    for j = 1:5
        newFolder = sprintf('Fold%d', j);
        cd(newFolder);
        load('cv-train.mat');
        load('cv-test.mat');
        X = cv_train(:,1:2);
        Y = cv_train(:,3);
        X_test = cv_test(:,1:2);
        Y_test = cv_test(:,3);
        for k = 1:size(C, 2)
            SVMModel = fitcsvm(X, Y, 'BoxConstraint', C(k),...
                'KernelFunction', 'RBF', 'KernelScale', sig);
            labels_test = predict(SVMModel, X_test);
            % error(i, j, k) = sum((labels_train - Y_test).^2)/(numel(Y_test));
            % error(i,j,k) = sum(labels_test ~= Y_test)/(numel(Y_test));
            error(i,j,k) = classification_error(labels_test, Y_test);
        end
        cd .. % return to CrossValidation from Foldi
    end
end
error_av_CV = mean(error, 2); % this is the cross-validation error, mean over folds
error_av_CV = squeeze(error_av_CV); % remove the empty dimension
[~, idx] = min(error_av_CV,[],2); % each row is a sigma, each column a C

%% Use chosen C to train and test
% clearvars -except sigma C error_av_CV idx

cd .. % return to Synthetic
load('train.mat');
load('test.mat');
cd .. % return to hw4-kit
X = train(:,1:2);
Y = train(:,3);
X_test = test(:,1:2);
Y_test = test(:,3);

train_error = zeros(1, size(sigma, 2));
test_error = zeros(1, size(sigma, 2));
CV_error = zeros(1, size(sigma, 2));

for i = 1:size(sigma, 2)
    sig = sigma(i);
    SVMModel = fitcsvm(X, Y, 'BoxConstraint', C(idx(i)),...
                    'KernelFunction', 'RBF', 'KernelScale', sig);
    labels_train = predict(SVMModel, X); % for training error
    % train_error(i) = sum((labels_train - Y).^2)/(numel(Y));
    % train_error(i) = sum(labels_train ~= Y)/(numel(Y));
    train_error(i) = classification_error(labels_train, Y);

    labels_test = predict(SVMModel, X_test); % for testing error
    % test_error(i) = sum((labels_test - Y_test).^2)/(numel(Y_test));
    % test_error(i) = sum(labels_test ~= Y_test)/(numel(Y_test));
    test_error(i) = classification_error(labels_test, Y_test);
    
    CV_error(i) = error_av_CV(i, idx(i));
    
    decision_boundary_SVM(X_test, Y_test, SVMModel, 1000, ...
        sprintf('Gaussian (sigma = %0.2f)', sigma(i)));
end

figure;
plot(log(sigma), CV_error, 'r--', ...
    log(sigma), train_error, 'g', ...
    log(sigma), test_error, 'b');
title('Sigma vs. Classification Error');
xlabel('log(sigma)');
ylabel('Classification Error');
legend('CV Error', 'Training Error', 'Test Error');

%% Helper Function

function err = classification_error(y_pred, y_true)
% This function computes the classification error for the predicted labels
% with respect to the ground truth. The returned error value is a real number
% between 0 and 1 (fraction of misclassications).

% y_true: vector of true labels (each label +1/-1)
% y_pred: vector of predicted labels (each prediction +1/-1)
% err: classification error (fraction of misclassifications)

	err = 1 - length(find(y_pred == y_true)) / length(y_true);
end