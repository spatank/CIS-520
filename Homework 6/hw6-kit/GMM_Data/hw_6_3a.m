clc; close all; clear;

normalized_llh_train = zeros(1,9);
normalized_llh_test = zeros(1,9);
K = 3;

% start in GMM_Data
for i = 1:9
    % train data
    cd TrainSubsets
    filename = sprintf('X_0.%d.mat', i);
    load(filename); % gives 'X'
    cd .. % return to GMM_Data
    cd MeanInitialization/Part_a
    filename_means = sprintf('mu_0.%d.mat', i); 
    load(filename_means); % gives 'mu'
    cd ../../ % return to GMM_Data
    [phis,mu,sigma] = em_algorithm(X,K,mu);
    normalized_llh_train(i) = compute_nllh(X,K,mu,sigma,phis);
    % test data
    load('X_test.mat');
    normalized_llh_test(i) = compute_nllh(X_test,K,mu,sigma,phis);
    
    sample_plot(sigma,mu,X_test,i)
end
cd TrainSubsets
load('X_1.mat'); % gives 'X'
cd .. % return to GMM_Data
cd MeanInitialization/Part_a
load('mu_1.mat'); % gives 'mu'
cd ../../ % return to GMM_Data
[phis,mu,sigma] = em_algorithm(X,K,mu);
normalized_llh_train(10) = compute_nllh(X,K,mu,sigma,phis);
normalized_llh_test(10) = compute_nllh(X_test,K,mu,sigma,phis);

sample_plot(sigma,mu,X_test,10)

percentage = 100.*[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 1];
figure;
hold on
plot(percentage, normalized_llh_train, 'r');
plot(percentage, normalized_llh_test, 'b');
hold off
title('Normalized Log-Likelihood vs. Percentage of Training Data Used','FontSize',15);
xlabel('Percentage of Training Data Used','FontSize',15);
ylabel('Normalized Log-Likelihood','FontSize',15);
legend('training','testing','Location','southeast');
