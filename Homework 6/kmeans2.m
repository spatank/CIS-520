clc; close all; clear;

load('k_means_data.mat');

curr_centroids1 = init_centroids_1;
curr_centroids2 = init_centroids_2;

for i = 1:5
    [idx,C] = kmeans(X,3,'MaxIter',1,...
        'Start',curr_centroids1);

    figure;
    subplot(2,1,1)
    grid on;
    xlim([0 20]);
    ylim([0 10]);
    set(gca,'XTick',0:1:20);
    set(gca,'YTick',0:1:10);
    hold on
    plot(X(idx==1,1),X(idx==1,2),'r*','MarkerSize',12);
    plot(X(idx==2,1),X(idx==2,2),'g*','MarkerSize',12);
    plot(X(idx==3,1),X(idx==3,2),'b*','MarkerSize',12);
    plot(curr_centroids1(:,1),curr_centroids1(:,2),'kx',...
        'MarkerSize',15,'LineWidth',3) ;
    legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
        'Location','best');
    titletext1 = sprintf('Iteration %d: Cluster Assignments', i);
    title(titletext1,'FontSize',15);
    hold off
    curr_centroids1 = C;

    subplot(2,1,2)
    grid on;
    xlim([0 20]);
    ylim([0 10]);
    set(gca,'XTick',0:1:20);
    set(gca,'YTick',0:1:10);
    hold on
    plot(X(idx==1,1),X(idx==1,2),'r*','MarkerSize',12);
    plot(X(idx==2,1),X(idx==2,2),'g*','MarkerSize',12);
    plot(X(idx==3,1),X(idx==3,2),'b*','MarkerSize',12);
    plot(curr_centroids1(:,1),curr_centroids1(:,2),'kx',...
        'MarkerSize',15,'LineWidth',3) ;
    % legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
    %     'Location','best');
    titletext2 = sprintf('Iteration %d: Centroid Updates', i);
    title(titletext2,'FontSize',15);
    hold off
end
    
for i = 1:6
    [idx,C] = kmeans(X,3,'MaxIter',1,...
        'Start',curr_centroids2);

    figure;
    subplot(2,1,1)
    grid on;
    xlim([0 20]);
    ylim([0 10]);
    set(gca,'XTick',0:1:20);
    set(gca,'YTick',0:1:10);
    hold on
    plot(X(idx==1,1),X(idx==1,2),'r*','MarkerSize',12);
    plot(X(idx==2,1),X(idx==2,2),'g*','MarkerSize',12);
    plot(X(idx==3,1),X(idx==3,2),'b*','MarkerSize',12);
    plot(curr_centroids2(:,1),curr_centroids2(:,2),'kx',...
        'MarkerSize',15,'LineWidth',3) ;
    legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
        'Location','best');
    titletext3 = sprintf('Iteration %d: Cluster Assignments', i);
    title(titletext3,'FontSize',15);
    hold off
    curr_centroids2 = C;
    
    subplot(2,1,2)
    grid on;
    xlim([0 20]);
    ylim([0 10]);
    set(gca,'XTick',0:1:20);
    set(gca,'YTick',0:1:10);
    hold on
    plot(X(idx==1,1),X(idx==1,2),'r*','MarkerSize',12);
    plot(X(idx==2,1),X(idx==2,2),'g*','MarkerSize',12);
    plot(X(idx==3,1),X(idx==3,2),'b*','MarkerSize',12);
    plot(curr_centroids2(:,1),curr_centroids2(:,2),'kx',...
        'MarkerSize',15,'LineWidth',3) ;
    % legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
    %     'Location','best');
    titletext4 = sprintf('Iteration %d: Centroid Updates', i);
    title(titletext4,'FontSize',15);
    hold off
end