function [labels] = kernel_regression(Xtrain, Ytrain, Xtest, sigma)

% kernel_regression - Evaluates kernel regression/classification predictions given training data and parameters.
% Function that implements kernel regression on the given data (binary classification)
% Usage: labels = kernel_regression(Xtrain,Ytrain,Xtest)
    
% Xtrain : N x P Matrix of training data, where N is the number of
%   training examples, and P is the dimensionality (number of features)
% Ytrain : N x 1 Vector of training labels (0/1)
% Xtest : M x P Matrix of testing data, where M is the number of
%   testing examples.
% sigma : width of the (gaussian) kernel.
% labels : return an M x 1 vector of predicted labels for testing data.
%
%   Returns an M x 1 vector that is the weighted average of the training labels of
%   according to a guassian kernel with width K and the given distance
%   function. Note that it is up to you to interpret these averages as
%   either the sign of the classification (for binary classifiation) or the
%   average prediction (for regression).

distFunc = 'l2';

% NOTE: this code is heavily VECTORIZED, which means that it does not use a
% any "for" loops and runs very quickly. Understanding this code is a
% good exercise for learning how to write programs in Matlab that run very
% fast.

numTestPoints = size(Xtest, 1);
numTrainPoints = size(Xtrain, 1);

% The following lines compute the difference between every test point and
% every train point in each dimension separately, using a single M x P X N
% 3-D array subtraction:

% Step 1:  Reshape the N x P training matrix into a 1 X P x N 3-D array
trainMat = reshape(Xtrain', [1 size(Xtrain,2) numTrainPoints]);
% Step 2:  Replicate the training array for each test point (1st dim)
trainCompareMat = repmat(trainMat, [numTestPoints 1 1]);
% Step 3:  Replicate the test array for each training point (3rd dim)
testCompareMat = repmat(Xtest, [1 1 numTrainPoints]);
% Step 4:  Element-wise subtraction
diffMat = testCompareMat - trainCompareMat;

% Now we can compute the distance functions on these element-wise
% differences:
if strcmp(distFunc, 'l2')
    distMat = sqrt(sum(diffMat.^2, 2));
elseif strcmp(distFunc, 'l1')
    distMat = sum(abs(diffMat), 2);
elseif strcmp(distFunc, 'linf')
    distMat = max(abs(diffMat), [], 2);
else
    error('Unrecognized distance function');
end

% Now we have a M x 1 x N 3-D array of distances between each pair of
% points. We squeeze this to a M x N matrix, then use these distances to 
% compute the corresponding M x N kernel matrix:

distMat = squeeze(distMat);
if numTestPoints == 1 % squeeze will make this a column vector if only 1 point
    distMat = distMat';
end

kernMat = exp(-distMat.^2/sigma.^2);

% Next, replicate the training label matrix to become M x N:
Ytrain = repmat(Ytrain', numTestPoints, 1);
% Finally, compute a weighted average over the M rows using the kernel:
labels = sum(Ytrain.*kernMat,2)./sum(kernMat,2);
end