clc; close all; clear;
load('data.mat');
X = new_data.X;
Y = new_data.Y;
lambda = 1;
w0 = zeros(3,1);

fun_l2 = @(w) norm((Y - X*w),2)^2 + lambda*norm(w,2)^2;
w_l2 = fminsearch(fun_l2,w0);

fun_l1 = @(w) norm((Y - X*w),2)^2 + lambda*norm(w,1);
w_l1 = fminsearch(fun_l1,w0);

% fun_l0 = @(w) norm((Y - X*w),2)^2 + lambda*sum(w ~= 0);
% w_l0 = fminsearch(fun_l0,w0);

%% Case 1
% [0; 0; 0]
% [1; 0; 0]
% [0; 1; 0]
% [0; 0; 1]
% [1; 1; 0]
% [1; 0; 1]
% [0; 1; 1]
% [1; 1; 1]
X_choice = X(:,1);
w0 = 0;
fun_l0 = @(w) norm((Y - X_choice*w),2)^2 + 1;
w_l0_est2 = fminsearch(fun_l0,w0);
w_l0_est2 = [w_l0_est2; 0; 0];
loss2 = norm((Y - X*w_l0_est2),2)^2;

X_choice = X(:,2);
w0 = 0;
fun_l0 = @(w) norm((Y - X_choice*w),2)^2 + 1;
w_l0_est3 = fminsearch(fun_l0,w0);
w_l0_est3 = [0; w_l0_est3; 0];
loss3 = norm((Y - X*w_l0_est3),2)^2;

X_choice = X(:,3);
w0 = 0;
fun_l0 = @(w) norm((Y - X_choice*w),2)^2 + 1;
w_l0_est4 = fminsearch(fun_l0,w0);
w_l0_est4 = [0; 0; w_l0_est4];
loss4 = norm((Y - X*w_l0_est4),2)^2;

X_choice = X(:,[1:2]);
w0 = [0; 0];
fun_l0 = @(w) norm((Y - X_choice*w),2)^2 + 2;
w_l0_est5 = fminsearch(fun_l0,w0);
w_l0_est5 = [w_l0_est5; 0];
loss5 = norm((Y - X*w_l0_est5),2)^2;

X_choice = [X(:,1),X(:,3)];
w0 = [0; 0];
fun_l0 = @(w) norm((Y - X_choice*w),2)^2 + 2;
w_l0_est6 = fminsearch(fun_l0,w0);
w_l0_est6 = [w_l0_est6(1); 0; w_l0_est6(2)];
loss6 = norm((Y - X*w_l0_est6),2)^2;

X_choice = [X(:,2),X(:,3)];
w0 = [0; 0];
fun_l0 = @(w) norm((Y - X_choice*w),2)^2 + 2;
w_l0_est7 = fminsearch(fun_l0,w0);
w_l0_est7 = [0; w_l0_est7];
loss7 = norm((Y - X*w_l0_est7),2)^2;

X_choice = X;
w0 = [0; 0; 0];
fun_l0 = @(w) norm((Y - X_choice*w),2)^2 + 3;
w_l0_est8 = fminsearch(fun_l0,w0);
loss8 = norm((Y - X*w_l0_est8),2)^2;

loss1 = norm((Y),2)^2;

loss = [loss1 loss2 loss3 loss4 loss5 loss6 loss7 loss8];

%% Ratios
w_MLE = inv(X'*X)*X'*Y;
ratio = norm(w_MLE,2)^2 / norm(Y - X*w_MLE,2)^2;
lambda = 16; % 6 (d)
% lambda = 3; 
w0 = zeros(3,1);

fun_l2 = @(w) norm((Y - X*w),2)^2 + lambda*norm(w,2)^2;
w_hat = fminsearch(fun_l2,w0);

ratio = norm(w_hat,2)^2 / norm(w_MLE,2)^2;
disp(ratio)

