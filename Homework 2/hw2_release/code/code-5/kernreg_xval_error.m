function [error] = kernreg_xval_error(X, Y, sigma, part, distFunc)
% KERNREG_XVAL_ERROR - Kernel regression cross-validation error.
%
% Usage:
%
%   ERROR = kernreg_xval_error(X, Y, SIGMA, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the kernel regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance function 
% to be used.
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KERNEL_REGRESSION

% [labels] = kernel_regression(Xtrain, Ytrain, Xtest, sigma)

% FILL IN YOUR CODE HERE

N = max(part); % number of folds in data
fold_error = zeros(1,N);

for i = 1:N
    X_test = X(part == i, :); % hold-out the i-th fold
    
    X_train = X(part ~= i, :); % remaining folds: X values
    Y_train = Y(part ~= i, :); % remaining folds: Y values
    
    [labels] = kernel_regression(X_train, Y_train, X_test, sigma);
    fold_error(i) = sum(labels ~= Y(part == i))/numel(Y(part == i));
end

error = mean(fold_error);

end