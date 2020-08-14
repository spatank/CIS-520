function [phis, mu, sigma] = em_algorithm(X_full,K,mu)

    n = size(X_full, 1); % number of data points
    d = size(X_full, 2); % number of features

    sigma = cell(K,1); % each row corresponds to a Gaussian
    sigma(cellfun('isempty',sigma)) = {eye(d,d)};
    phis = (1/K).*ones(1,K); % each column corresponds to a Gaussian
    gammas = zeros(n,K);
    llh = compute_nllh(X_full,K,mu,sigma,phis);
    for t = 1:1000
            for j = 1:K
                numerator = phis(j) * gaussian_pdf(X_full,mu(j,:),sigma{j});
                denominator = 0;
                for m = 1:K
                    denominator = denominator + ...
                        phis(m)*gaussian_pdf(X_full,mu(m,:),sigma{m});
                end
                gammas(:,j) = numerator ./ denominator;
                N_k = sum(gammas(:,j));
                cov_new = zeros(d,d);
                mu_new = zeros(1,d);
                for i = 1:n
                    mu_new = mu_new + (gammas(i,j) .* (X_full(i,:)));
                end
                mu(j,:) = (1/N_k).*mu_new;
                for i = 1:n
                    cov_new = cov_new + ...
                        gammas(i,j).*(X_full(i,:)-mu(j,:))'*(X_full(i,:)-mu(j,:));
                end
                sigma{j} = (1/N_k).*cov_new;
                phis(j) = N_k/n;
            end
            llh_new = compute_nllh(X_full,K,mu,sigma,phis);
            if (llh_new - llh) < 10^-6
                fprintf('Broke Here at t = %d\n', t);
                break;
            else
                llh = llh_new;
            end
    end
    
end

