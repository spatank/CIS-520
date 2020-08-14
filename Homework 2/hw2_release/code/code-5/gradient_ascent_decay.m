function [weights,error_per_iter] = gradient_ascent_decay(Xtrain,Ytrain,initial_step_size,iterations)

    % Function to perform gradient ascent with a decaying step size for
    % logistic regression.
    % Usage: [weights,error_per_iter] = gradient_ascent(Xtrain,Ytrain,step_size,iterations)
    
    % The parameters to this function are exactly the same as the
    % parameters to gradient ascent with fixed step size.
    
    % initial_step_size : This parameter refers to the initial value of the step
    % size. The actual step size to update the weights will be a value
    % that is (initial_step_size * some function that decays over time)
    % some good choices for this function might by 1/n or 1/sqrt(n).
    % Experiment with such functions, and initial step size until you get
    % good performance.
    
    weights = ones(size(Xtrain,2),1); % P x 1 vector of initial weights
    error_per_iter = zeros(iterations,1); % error_per_iter(i) records training error in iteration i of GD.
    % dont forget to update these values within the loop!
    
    % FILL IN THE REST OF THE CODE %
    
    step = initial_step_size;
    
    for iter = [1:iterations]
        % computed the gradient
        intermediate = zeros(size(Xtrain)); % for each feature
        for i = 1:size(Xtrain, 1)
            % check divide by N
            % intermediate = (Ytrain(i).*Xtrain(i,:)) - Xtrain(i,:)/(1 + exp(-(weights').*(Xtrain(i,:))));
            intermediate(i,:) = (Ytrain(i).*Xtrain(i,:)) - Xtrain(i,:)/(1 + exp(-(Xtrain(i,:)*weights)));
        end
        
        % took a step in the direction of the gradient
        gradient = sum(intermediate, 1); %1 x P vector
        weights = weights + step*gradient';
        
        % calculated residual
        % error_per_iter(iter) = sum(Xtrain*weights - Ytrain)^2/size(Xtrain,1);
        
        
        % calculated residual
        % error_per_iter(iter) = sum(Xtrain*weights - Ytrain)^2/size(Xtrain,1);
        labels = sign(Xtrain*weights);
        labels(labels < 0) = 0;
        labels(labels > 0) = 1;
        error_per_iter(iter) = sum((labels ~= Ytrain))/numel(Ytrain);
        
        % TODO: consider other functions
        step = 1/sqrt(step);
    end

end

