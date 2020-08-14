clc; clear; close all;
C = [1, 10, 10^2, 10^3, 10^4, 10^5];
error = zeros(5, size(C, 2));
cd Synthetic/CrossValidation
sig = 0.1;
% make sure to be in the CrossValidation folder for the loop to work
for i = 1:5
    newFolder = sprintf('Fold%d', i);
    cd(newFolder);
    load('cv-train.mat');
    load('cv-test.mat');
    X = cv_train(:,1:2);
    Y = cv_train(:,3);
    X_test = cv_test(:,1:2);
    Y_test = cv_test(:,3);
    for j = 1:size(C, 2)
        SVMModel = fitcsvm(X, Y, 'BoxConstraint', C(j),...
            'KernelFunction', 'RBF', 'KernelScale', sig);
        labels_train = predict(SVMModel, X_test);
        % error(i,j) = sum((labels_train - Y_test).^2)/(numel(Y_test));
        error(i,j) = sum(labels_train ~= Y_test)/(numel(Y_test));
    end
    cd .. % return to CrossValidation from Foldi
end
error_av_CV = mean(error, 1); % this is the cross-validation error
% 10^4 looks to have the smallest CV error
[~, idx] = min(error_av_CV);
C_choice = C(idx);
cd ..
cd ..