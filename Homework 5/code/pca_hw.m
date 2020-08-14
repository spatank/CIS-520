clc; close all; clear;

load('MNIST_train.mat')

mean_column = mean(X_train, 1);
mean = repmat(mean_column, 12000, 1);
mc_X_train = X_train - mean;

% testImage = imrotate(fliplr(reshape(mc_X_train(end,:), 28, 28)),90);
% figure;
% colormap(gray);
% imagesc(testImage)

[loadings, scores, latent, tsquared, explained] = pca(mc_X_train, 'Centered', false);

% testImagePCA = imrotate(fliplr(reshape(loadings(:,1), 28, 28)),90);
% figure;
% colormap(gray);
% imagesc(testImagePCA)
% title('1st PC Vector as an Image');
% 
% testImagePCA = imrotate(fliplr(reshape(loadings(:,2), 28, 28)),90);
% figure;
% colormap(gray);
% imagesc(testImagePCA)
% title('2nd PC Vector as an Image');
% 
% testImagePCA = imrotate(fliplr(reshape(loadings(:,3), 28, 28)),90);
% figure;
% colormap(gray);
% imagesc(testImagePCA)
% title('3rd PC Vector as an Image');
% 
% testImagePCA = imrotate(fliplr(reshape(loadings(:,784), 28, 28)),90);
% figure;
% colormap(gray);
% imagesc(testImagePCA)
% title('784th PC Vector as an Image');

%% First 2 PCs for Digit 1 and 8

% idx_0 = Y_train == 1;
% scores0 = scores(idx_0, :);
% idx_7 = Y_train == 8;
% scores7 = scores(idx_7, :);
% 
% figure;
% scatter(scores0(:,1), scores0(:,2), 'r')
% hold on;
% scatter(scores7(:,1), scores7(:,2), 'b');
% hold off;
% legend('0', '7');
% xlabel('PC 1');
% ylabel('PC 2');
% 
% figure;
% scatter(scores0(:,100), scores0(:,101), 'r')
% hold on;
% scatter(scores7(:,100), scores7(:,101), 'b');
% hold off;
% legend('0', '7');
% xlabel('PC 100');
% ylabel('PC 101');

%% Accuracy
% frac_acc = cumsum(explained)';
% figure;
% plot(1:1:784, frac_acc);
% title('Fractional Reconstruction Accuracy');
% xlabel('Principal Components Included');
% ylabel('Fractional Accuracy (%)');

%% Reconstructions

% pcs_needed = [784 264 88 43 25 15 9 6 4 2];
% recon_accuracy = [100 90 80 70 60 50 40 30 20 10];
% 
% for i = 1:numel(pcs_needed)
%     reconstruct = (scores(500,1:pcs_needed(i))*loadings(:,1:pcs_needed(i))') + mean_column;
%     testRecon = imrotate(fliplr(reshape(reconstruct, 28, 28)),90) ;
%     subplot(3,4,i)
%     colormap(gray);
%     imagesc(testRecon)
%     title_text = sprintf('%d%% Reconstruction', recon_accuracy(i));
%     title(title_text);
% end
% 
% figure;
% for i = 1:numel(pcs_needed)
%     reconstruct = (scores(6000,1:pcs_needed(i))*loadings(:,1:pcs_needed(i))') + mean_column;
%     testRecon = imrotate(fliplr(reshape(reconstruct, 28, 28)),90) ;
%     subplot(3,4,i)
%     colormap(gray);
%     imagesc(testRecon)
%     title_text = sprintf('%d%% Reconstruction', recon_accuracy(i));
%     title(title_text);
% end
% 
% figure;
% for i = 1:numel(pcs_needed)
%     reconstruct = (scores(10000,1:pcs_needed(i))*loadings(:, 1:pcs_needed(i))') + mean_column;
%     testRecon = imrotate(fliplr(reshape(reconstruct, 28, 28)),90) ;
%     subplot(3,4,i)
%     colormap(gray);
%     imagesc(testRecon)
%     title_text = sprintf('%d%% Reconstruction', recon_accuracy(i));
%     title(title_text);
% end

%% Autoencoders

% 30% accuracy needs 6 PCs
rng('default');
autoenc30 = trainAutoencoder520(mc_X_train', 6, 'MaxEpochs',250,...
'LossFunction','mse');
rng('default');
% 60% accuracy needs 25 PCs
autoenc60 = trainAutoencoder520(mc_X_train', 25, 'MaxEpochs',250,...
'LossFunction','mse');
rng('default');
% 90% accuracy needs 264 PCs
autoenc90 = trainAutoencoder520(mc_X_train', 264, 'MaxEpochs',250,...
'LossFunction','mse');

% 30% accuracy needs 6 PCs
rng('default');
lin_autoenc30 = trainAutoencoder520(mc_X_train', 6, 'MaxEpochs',250,...
'EncoderTransferFunction','purelin','DecoderTransferFunction','purelin',...
'LossFunction','mse');
rng('default');
% 60% accuracy needs 25 PCs
lin_autoenc60 = trainAutoencoder520(mc_X_train', 25, 'MaxEpochs',250,...
'EncoderTransferFunction','purelin','DecoderTransferFunction','purelin',...
'LossFunction','mse');
rng('default');
% 90% accuracy needs 264 PCs
lin_autoenc90 = trainAutoencoder520(mc_X_train', 264, 'MaxEpochs',250,...
'EncoderTransferFunction','purelin','DecoderTransferFunction','purelin',...
'LossFunction','mse');

