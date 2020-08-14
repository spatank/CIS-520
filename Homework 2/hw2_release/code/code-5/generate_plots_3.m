
clc; close all; clear;
load('Y.mat');
load('X.mat');
% load('X_noisy.mat');
% X = X_noisy;

iterations = 850; 
stepsize = 0.0001; 


[~, error_per_iter] = gradient_ascent_fixed(X,Y,stepsize,iterations);

clear weights
X = [X, ones(size(X,1),1)];
[weights, error_per_iter_cons] = gradient_ascent_fixed(X,Y,stepsize,iterations);

figure;
plot(1:iterations, error_per_iter, 'r', 1:iterations, error_per_iter_cons);
xlabel('Iterations');
ylabel('Error');
legend('Without Constant Feature', 'With Constant Feature');
title('Error vs. Iterations');