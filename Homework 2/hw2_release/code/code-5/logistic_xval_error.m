function [error] = logistic_xval_error(X, Y, part)
% LOGISTIC_XVAL_ERROR - Logistic regression cross-validation error.
%
% Usage:
%
%   ERROR = logistic_xval_error(X, Y, PART)
%
% Returns the average N-fold cross validation error of the logistic regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION).
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, LOGISTIC_REGRESSION

% FILL IN YOUR CODE HERE

N = max(part); % number of folds in data
fold_error = zeros(N,1);
stepsize = 0.0001; % TODO: check!!
iterations = 750; % TODO: check!!

for i = 1:N
    X_test = X(part == i, :); % hold-out the i-th fold
    
    X_train = X(part ~= i, :); % remaining folds: X values
    Y_train = Y(part ~= i, :); % remaining folds: Y values
    
    [labels] = logistic_regression(X_train, Y_train, X_test, stepsize, iterations);
    fold_error(i) = (sum(labels ~= Y(part == i))/numel(Y(part == i)));
end

error = mean(fold_error);

end