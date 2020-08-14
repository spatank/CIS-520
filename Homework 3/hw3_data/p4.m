clc; clear; close all;
load('train_data.mat');
load('train_y.mat');
x1 = X(:,1);
x2 = X(:,2);
x3 = X(:,3);
n = size(X,1);

%% Model 1: with one feature only: x_1
X = x1;
w1 = inv(X'*X)*X'*Y; % weights using Moore-Penrose Pseudo-Inverse
Y_pred = X * w1;
SSE1 = sum((Y_pred - Y).^2);
ERR_bits_1 = n*(log(SSE1/n)/log(2));
% ERR_bits_1 = n*(log(SSE1/n));
AIC_model_1 = 2; % 2 * 1 bit/feature, 1 feature
BIC_model_1 = 1 * 2 * (0.5)*(log(n)/log(2)); 
AIC_bits_1 = ERR_bits_1 + AIC_model_1;
BIC_bits_1 = ERR_bits_1 + BIC_model_1;
%% Model 2: with two features: x_1 and x_2
X = [x1,x2];
w2 = inv(X'*X)*X'*Y; % weights using Moore-Penrose Pseudo-Inverse
Y_pred = X * w2;
SSE2 = sum((Y_pred - Y).^2);
ERR_bits_2 = n*(log(SSE2/n)/log(2));
AIC_model_2 = 4; % 2 * 1 bit/feature, 1 feature
BIC_model_2 = 2 * 2 * (0.5)*(log(n)/log(2)); 
AIC_bits_2 = ERR_bits_2 + AIC_model_2;
BIC_bits_2 = ERR_bits_2 + BIC_model_2;
%% Model 3: with three features: x_1, x_2, x_3
X = [x1,x2,x3];
w3 = inv(X'*X)*X'*Y; % weights using Moore-Penrose Pseudo-Inverse
Y_pred = X * w3;
SSE3 = sum((Y_pred - Y).^2);
ERR_bits_3 = n*(log(SSE3/n)/log(2));
AIC_model_3 = 6; % 2 * 1 bit/feature, 1 feature
BIC_model_3 = 3 * 2 * (0.5)*(log(n)/log(2)); 
AIC_bits_3 = ERR_bits_3 + AIC_model_3;
BIC_bits_3 = ERR_bits_3 + BIC_model_3;