clc; close all; clear;

%% 2 (a)

load('breast_cancer.mat'); % gives X_train and Y_train
X = X_train;
Y = Y_train;
Z = inv(X'*X)^(1/2) * X'*Y * inv(sum(Y.*Y))^(1/2);
[U,S,V] = svd(Z);
XU = X*U(:,1);
YV_T = Y*V';
R_a = corrcoef(XU,YV_T);

%% 2 (b)

[loadings, scores] = pca(X_train);
x_pca = scores(:,1); % scores of first PC
w = sum(x_pca .* Y_train)/sum(x_pca.^2);
Y_hat = w * x_pca;
R_b = corrcoef(Y_train,Y_hat);