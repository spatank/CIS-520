clc; close all; clear;

K = [1, 2, 3, 4, 5];
normalized_llh_train = zeros(1,numel(K));
% normalized_llh_CV = zeros(1,numel(K));
normalized_llh_test = zeros(1,numel(K));
normalized_llh = zeros(5,5);

% start in GMM_Data
for i = 1:5 % loop over folders
    cd CrossValidation
    foldername = sprintf('Fold%d', i);
    cd(foldername)
    load('cv-train.mat'); % gives 'X_train;
    load('cv-test.mat'); % gives 'X_test;
    cd ../../ % return to GMM_Data
    for j = 1:5 % loop over Ks
        k = K(j);
        cd MeanInitialization/Part_b
        filename = sprintf('mu_k_%d.mat', j);
        load(filename); % gives 'mu'
        cd ../../ % return to GMM_Data
        [phis,mu_CV,sigma] = em_algorithm(X_train,k,mu);
        normalized_llh(i,j) = compute_nllh(X_test,k,mu_CV,sigma,phis);
        
        % train data
        load('X.mat'); % gives 'X_full'
        [phis,mu_train,sigma] = em_algorithm(X_full,k,mu);
        normalized_llh_train(j) = compute_nllh(X_full,k,mu_train,sigma,phis);
        
        % test data
        load('X_test.mat'); % gives 'X_test'
        normalized_llh_test(j) = compute_nllh(X_test,k,mu_train,sigma,phis);
    end
end
normalized_llh_CV = mean(normalized_llh,1);
figure;
hold on
plot(K, normalized_llh_CV, 'r');
plot(K, normalized_llh_test, 'g');
plot(K, normalized_llh_train, 'b');
hold off
title('Normalized Log-Likelihood vs. Number of Gaussians in Mixture','FontSize',15);
xlabel('Number of Gaussians in Mixture','FontSize',15);
ylabel('Normalized Log-Likelihood','FontSize',15);
legend('CV','testing','training','Location','southeast');