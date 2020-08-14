clc; close all; clear;

load('k_means_data.mat');

[idx,C] = kmeans(X,3,'Display','iter','MaxIter',5,...
    'Start',init_centroids_1);
figure;
grid on;
xlim([0 20]);
ylim([0 10]);
set(gca,'XTick',0:1:20);
set(gca,'YTick',0:1:10);
% pbaspect([2 1 1])
hold on
plot(X(idx==1,1),X(idx==1,2),'r*','MarkerSize',12);
plot(X(idx==2,1),X(idx==2,2),'g*','MarkerSize',12);
plot(X(idx==3,1),X(idx==3,2),'b*','MarkerSize',12);
plot(C(:,1),C(:,2),'kx',...
    'MarkerSize',15,'LineWidth',3) ;
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
    'Location','best');
title('Cluster Assignments and Centroids for Part (a)');
hold off
    
[idx,C] = kmeans(X,3,'Display','iter','MaxIter',5,...
    'Start',init_centroids_2);
figure;
grid on
xlim([0 20]);
ylim([0 10]);
set(gca,'XTick',0:1:20);
set(gca,'YTick',0:1:10);
% pbaspect([2 1 1])
hold on
plot(X(idx==1,1),X(idx==1,2),'r*','MarkerSize',12);
plot(X(idx==2,1),X(idx==2,2),'g*','MarkerSize',12);
plot(X(idx==3,1),X(idx==3,2),'b*','MarkerSize',12);
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) ;
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
    'Location','best');
title('Cluster Assignments and Centroids for Part (b)');
hold off