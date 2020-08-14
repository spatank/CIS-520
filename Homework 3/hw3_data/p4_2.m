clc; clearvars -except SSE1 SSE2 SSE3 w1 w2 w3; close all;
load('test_data.mat');
load('test_y.mat');
x1 = Xtest_new(:,1);
x2 = Xtest_new(:,2);
x3 = Xtest_new(:,3);
Y = Ytest_new;
n = size(Xtest_new,1);

%% Model 1: with one feature only: x_1
X = x1;
% w1 = inv(X'*X)*X'*Y; % weights using Moore-Penrose Pseudo-Inverse
Y_pred = X * w1;
test_SSE1 = sum((Y_pred - Y).^2);

%% Model 2: with two features: x_1 and x_2
X = [x1,x2];
% w2 = inv(X'*X)*X'*Y; % weights using Moore-Penrose Pseudo-Inverse
Y_pred = X * w2;
test_SSE2 = sum((Y_pred - Y).^2);

%% Model 3: with three features: x_1, x_2, x_3
X = [x1,x2,x3];
% w3 = inv(X'*X)*X'*Y; % weights using Moore-Penrose Pseudo-Inverse
Y_pred = X * w3;
test_SSE3 = sum((Y_pred - Y).^2);
