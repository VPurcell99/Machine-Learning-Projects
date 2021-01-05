%% *Vincent Purcell - HW 4 - ECE487*
clear; clc; close all;

%% *Problem 4.6*
% Problem 4.6 from the Text on page 248.

% Data generation based on inputs from text book
rng('default')
rng(1)
m = [-5 5 5 -5; 5 -5 5 -5];
s = 2;
N = 100;
[x1,y1] = data_generator(m,s,N);
x1 = x1';
y1 = y1';
rng(10);
[x2,y2] = data_generator(m,s,N);
x2 = x2';
y2 = y2';

C_vec = [1,100,1000]';
sigma_vec = [0.5,1,2,4]';
tol = 0.001;

% Create 12 models and plot them based on all combinations of sigma and C
for i=1:size(C_vec)
    for j=1:size(sigma_vec)
        plotSVM(x1,y1,x2,y2,tol,C_vec(i),sigma_vec(j));
    end
end

% Call Decision Tree Function
decisionTree(x1,y1,x2,y2);


%% SVM Classification

%%% Classification and Plot Function
function plotSVM(x1,y1,x2,y2,tol,C,sigma)

    %Get classifier model and errors
    [model, test_err, train_err] = SVM_clas(x1,y1,x2,y2,tol,C,sigma);
    svInd = model.IsSupportVector;
    %Below plotting methods adapted from fitcsvm MATLAB documentation
    h = 0.02;
    [X1,X2] = meshgrid(min(x1(:,1)):h:max(x1(:,1)),...
        min(x1(:,2)):h:max(x1(:,2)));
    [~,score] = predict(model,[X1(:),X2(:)]);
    scoreGrid = reshape(score(:,1),835,916);

    figure
    plot(x1(:,1),x1(:,2),'k.')
    hold on
    plot(x1(svInd,1),x1(svInd,2),'ko','MarkerSize',10)
    contour(X1,X2,scoreGrid)
    colorbar;
    title_str = "SVM Classification C=" + num2str(C) + " \sigma=" + num2str(sigma);
    title(title_str)
    xlabel('X Axis')
    ylabel('Y Axis')
    legend('Observation','Support Vector')
    a = gca; % get the current axis;
    % set the width of the axis (the third value in Position) 
    % to be 60% of the Figure's width
    a.Position(3) = 0.6;
    text1 = {"Error","Train = " + num2str(train_err) ...
        ,"Test = " + num2str(test_err)};
    annotation('textbox',[0.83 0 0 .5],'String',text1,'FitBoxToText','on')
    hold off
    snapnow
end

%%% SVM Classifier
% Function adapted from function on page 247 of the text
function [model,test_err,train_err]=SVM_clas(x1,y1,x2,y2,tol,C,sigma)

    % The following options are from the function in the textbook, it
    % required simple adaptation to the new function fitcsvm:
    % DeltaGradientTolerance = tol
    % Solver = SMO
    % Verbose = 1
    % IterationLimit = 20000
    % CacheSize = 10000
    % KernelFunction = RBF
    % KernelScale = sigma
    % BoxConstraint = C
    model = fitcsvm(x1,y1, ...
        'DeltaGradientTolerance',tol,...
        'Solver','SMO',...
        'Verbose',1,...
        'IterationLimit',20000,...
        'CacheSize',10000,...
        'KernelFunction','RBF',...
        'KernelScale',sigma,...
        'BoxConstraint',C);
    
    %Computation of the error probability 
    test_err = loss(model,x2,y2);
    train_err = loss(model,x1,y1);
end

%% *Decision Tree Classification*

function decisionTree(x1,y1,x2,y2)
    rng(1);
    tree = fitctree(x1, y1, 'Prune', 'off','PruneCriterion','impurity');
    tree_pruned = prune(tree);
    
    view(tree,'Mode','graph');
    view(tree_pruned,'Mode','graph');
    
    test_err = loss(tree,x2,y2);
    train_err = loss(tree,x1,y1);
    
    test_err_p = loss(tree_pruned,x2,y2);
    train_err_p = loss(tree_pruned,x1,y1);
    
    fprintf('Testing Error without Pruning:  %f\n', test_err);
    fprintf('Training Error without Pruning: %f\n', train_err);
    fprintf('Testing Error with Pruning:     %f\n', test_err_p);
    fprintf('Training Error with Pruning:    %f\n', train_err_p);
end

%% *Functions Received From Textbook*
%  The following functions were received from the Textbook
%  Pattern Recognition - Theodoridis, Koutroumbas

%%% Data Generation Class
% Received from page 244 of the text
function [x,y]=data_generator(m,s,N) 
    S = s*eye(2);
    [~,c] = size(m); 
    x = []; % Creating the training set 
    for i = 1:c 
        x = [x mvnrnd(m(:,i)',S,N)']; 
    end
    y=[ones(1,N) ones(1,N) -ones(1,N) -ones(1,N)];
end


















