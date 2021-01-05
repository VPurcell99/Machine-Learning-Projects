%% *Vincent Purcell - HW 7 - ECE487*
clear; clc; close all;

%% *Problem 6.1*
% Problem 6.1 from the Text on page 401.
X = generate_hyper([1;1],0,10,1,1000,0);
[pc,variances]=pcacov(cov(X'))
X_0 = pc(1,:)'.*X;

%Plot subplots
figure;
subplot(2,2,[1 3]); plot(X(1,:),X(2,:),'.b'); %X
title("Data");
xlabel("X_0");
ylabel("X_1");
xlim([-10 10])
ylim([-15 15])
subplot(2,2,[2 4]); plot(X_0(1,:),X_0(2,:),'.r'); %PCA
title("PCA");
xlabel("PC1");
ylabel("PC2");
xlim([-10 10])
ylim([-15 15])
sgtitle("Principal Component Analysis");

mdl(X)

%% *Functions Received From Textbook*
%  The following functions were received from the Textbook
%  Pattern Recognition - Theodoridis, Koutroumbas

%%% Generate Hyperplane Function
% Adapted from page 399 of the text
function X=generate_hyper(w,w0,a,e,N,sed) 
    rng(sed);
    l=length(w);
    t=(rand(l-1,N)-.5)*2*a; 
    t_last=-(w(1:l-1)/w(l))'*t + 2*e*(rand(1,N)-.5)-(w0/w(l)); 
    X=[t; t_last];
end