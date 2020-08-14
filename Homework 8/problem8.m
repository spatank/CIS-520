clc; close all; clear;

gamma = 0.9;

p_s_new_sa = zeros(5,5,2); 
r_s_new_sa = zeros(5,5,2);

% forward
p_s_new_sa(1,:,1) = [0.1, 0.9, 0, 0, 0];
p_s_new_sa(2,:,1) = [0.1, 0, 0.9, 0, 0];
p_s_new_sa(3,:,1) = [0.1, 0, 0, 0.9, 0];
p_s_new_sa(4,:,1) = [0.1, 0, 0, 0, 0.9];
p_s_new_sa(5,:,1) = [0.1, 0, 0, 0, 0.9];

r_s_new_sa(1,:,1) = [2, 0, 0, 0, 0];
r_s_new_sa(2,:,1) = [2, 0, 0, 0, 0];
r_s_new_sa(3,:,1) = [2, 0, 0, 0, 0];
r_s_new_sa(4,:,1) = [2, 0, 0, 0, 0];
r_s_new_sa(5,:,1) = [2, 0, 0, 0, 10];

% backward
p_s_new_sa(1,:,2) = [0.9, 0.1, 0, 0, 0];
p_s_new_sa(2,:,2) = [0.9, 0, 0.1, 0, 0];
p_s_new_sa(3,:,2) = [0.9, 0, 0, 0.1, 0];
p_s_new_sa(4,:,2) = [0.9, 0, 0, 0, 0.1];
p_s_new_sa(5,:,2) = [0.9, 0, 0, 0, 0.1];

r_s_new_sa(1,:,2) = [2, 0, 0, 0, 0];
r_s_new_sa(2,:,2) = [2, 0, 0, 0, 0];
r_s_new_sa(3,:,2) = [2, 0, 0, 0, 0];
r_s_new_sa(4,:,2) = [2, 0, 0, 0, 0];
r_s_new_sa(5,:,2) = [2, 0, 0, 0, 10];


%% Optimal State Value 

V_curr = zeros(5,1);
iters = 100000;

for i = 1:iters
    test1 = p_s_new_sa;
    test2 = r_s_new_sa + (gamma .* repmat(V_curr',5,1,2));
    V_curr_new = test1 .* test2;
    sum_over_states = squeeze(sum(V_curr_new,2));
    V_curr_new = max(sum_over_states,[],2);
    V_curr = V_curr_new;
end
disp(V_curr)


%% Optimal Policy

test1 = p_s_new_sa;
test2 = r_s_new_sa + (gamma .* repmat(V_curr',5,1,2));
V_curr_new = test1 .* test2;
sum_over_states = squeeze(sum(V_curr_new,2));
[maxVal, idx] = max(sum_over_states,[],2);
disp(idx);





