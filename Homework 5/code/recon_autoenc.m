clc; close all; clear;

load('workspace_hw5_2.mat')

sum_eigs = sum(latent);
% frac_acc = cumsum(explained)';
pcs_needed = fliplr([784 264 88 43 25 15 9 6 4 2]);
recon_accuracy = fliplr([100 90 80 70 60 50 40 30 20 10]);

% size of hidden state = 6
x_recon_30 = predict(autoenc30, mc_X_train');
recon_30_acc = 1 - (sum(sum((mc_X_train - x_recon_30').^2,2))/size(mc_X_train,1))/sum_eigs;

% size of hidden state = 25
x_recon_60 = predict(autoenc60, mc_X_train');
recon_60_acc = 1 - (sum(sum((mc_X_train - x_recon_60').^2,2))/size(mc_X_train,1))/sum_eigs;

% size of hidden state = 264
x_recon_90 = predict(autoenc90, mc_X_train');
recon_90_acc = 1 - (sum(sum((mc_X_train - x_recon_90').^2,2))/size(mc_X_train,1))/sum_eigs;

% size of hidden state = 6
x_lin_recon_30 = predict(lin_autoenc30, mc_X_train');
lin_recon_30_acc = 1 - (sum(sum((mc_X_train - x_lin_recon_30').^2,2))/size(mc_X_train,1))/sum_eigs;

% size of hidden state = 25
x_lin_recon_60 = predict(lin_autoenc60, mc_X_train');
lin_recon_60_acc = 1 - (sum(sum((mc_X_train - x_lin_recon_60').^2,2))/size(mc_X_train,1))/sum_eigs;

% size of hidden state = 264
x_lin_recon_90 = predict(lin_autoenc90, mc_X_train');
lin_recon_90_acc = 1 - (sum(sum((mc_X_train - x_lin_recon_90').^2,2))/size(mc_X_train,1))/sum_eigs;

hidden_size = [6 25 264];
recon_acc = 100.*[recon_30_acc recon_60_acc recon_90_acc];
lin_recon_acc = 100.*[lin_recon_30_acc lin_recon_60_acc lin_recon_90_acc];

figure;
hold on
% scatter(pcs_needed, recon_accuracy, 100, 'd');
% scatter(hidden_size, recon_acc, 100, 'x');
% scatter(hidden_size, lin_recon_acc, 100, 'o');
plot(pcs_needed(1:9), recon_accuracy(1:9), 'r--');
plot(hidden_size, recon_acc, 'b');
plot(hidden_size, lin_recon_acc, 'g');
hold off
title('Fractional Reconstruction Accuracy');
xlabel('Size of Hidden State / PCs Included');
ylabel('Fractional Accuracy (%)');
legend('PCA', 'Non-Linear Encoders', 'Linear Encoders', 'Location', 'southeast');

