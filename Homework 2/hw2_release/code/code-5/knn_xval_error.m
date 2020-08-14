function [error] = knn_xval_error(X, Y, K, part, distfunc)
% KNN_XVAL_ERROR - KNN cross-validation error.
%
% Usage:
%
%   ERROR = knn_xval_error(X, Y, K, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the K-NN algorithm on the 
% given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used.
%
% Note that N = max(PART), corresponding to the number of folds.
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, K_NEAREST_NEIGHBOURS

% FILL IN YOUR CODE HERE
% [labels] = k_nearest_neighbours(Xtrain, Ytrain, Xtest, K, distfunc);

N = max(part); % number of folds in data
fold_error = zeros(1,N);

for i = 1:N
    X_test = X(part == i, :); % hold-out the i-th fold
    
    X_train = X(part ~= i, :); % remaining folds: X values
    Y_train = Y(part ~= i, :); % remaining folds: Y values
    
    [labels] = k_nearest_neighbours(X_train, Y_train, X_test, K, distfunc);
    fold_error(i) = sum(labels ~= Y(part == i))/(numel(Y(part == i)));
end

error = mean(fold_error);