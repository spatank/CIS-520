function [part] = make_xval_partition(n, n_folds)

% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% YOUR CODE GOES HERE

% remainder
k = floor(n/n_folds);
remainder = mod(n, n_folds);

% non-remainder points shuffled
vec = randperm(n-remainder);
part = ceil(vec/k);

% remainder points shuffled
rem_vector = randperm(n_folds, remainder);

% combine non-remainder and remainder
part = [part, rem_vector];

