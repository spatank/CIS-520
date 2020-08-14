function [labels] = k_nearest_neighbours(Xtrain, Ytrain, Xtest, K, distfunc)
% KNN_TEST - Evaluates KNN predictions given training data and parameters.

% Function to implement the K nearest neighbours algorithm on the given
% dataset.
% Usage: labels = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K)
    
% Xtrain : N x P Matrix of training data, where N is the number of
%   training examples, and P is the dimensionality (number of features)
% Ytrain : N x 1 Vector of training labels (0/1)
% Xtest : M x P Matrix of testing data, where M is the number of
%   testing examples.
% K : number of nearest neighbours used to make predictions on the test
%     dataset. Remember to take care of corner cases.
% distfunc: distance function to be used - l1, l2, linf.
% labels : return an M x 1 vector of predicted labels for testing data.
%
%   Returns an M x 1 vector that is the average of the training labels of
%   the K nearest neighbors to each test point using the given distance
%   function. Note that it is up to you to interpret these averages as
%   either the sign of the classification (for binary classifiation) or the
%   average prediction (for regression).

if nargin < 5
    distfunc = 'l2';
end

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
if strcmp(distfunc, 'l2')
    distMat = sqrt(sum(diffMat.^2, 2));
elseif strcmp(distfunc, 'l1')
    distMat = sum(abs(diffMat), 2);
elseif strcmp(distfunc, 'linf')
    distMat = max(abs(diffMat), [], 2);
else
    error('Unrecognized distance function');
end

% Now we have a M x 1 x N 3-D array of distances between each pair of points.
% To find the nearest neighbors, we first "squeeze" this to become a M x N
% matrix, and then sort within each of the M rows separately. Note that we
% use only the second output from the "sort" command.

distMat = squeeze(distMat);
if numTestPoints == 1 % if only 1 point, squeeze converts to col vector
    distMat = distMat';
end

[~, nnMat] = sort(distMat, 2);

nnMat = nnMat(:,1:K);

% Average over the nearest neighbors to get predicted labels
labels = Ytrain(nnMat);
if size(nnMat, 1) == 1 % again, if only 1, gets incorrectly converted to col vector
    labels = labels';
end

if K > 1
    labels = mean(labels, 2);
    labels(labels > 0) = 1;
end


    
end