%% *Vincent Purcell - HW 6 - ECE487*
clear; clc; close all;

%% *Problem 5.1*
% Problem 5.1 from the Text on page 316.

rng(10);

%%% Part A
% mu1=0, mu2=2, N1=100, N2=100
N1 = 100;
N2 = 100;
mu1 = 0;
mu2 = 2;
var = 1;
runTtest(mu1,mu2,var,N1,N2,"Part A");

%%% Part B
% mu1=0, mu2=0.2, N1=100, N2=100
runTtest(mu1,0.2,var,N1,N2,"Part B");

%%% Part C
% mu1=0, mu2=2, N1=150, N2=250 for part 1 and
% mu1=0, mu2=0.2, N1=150, N2=250 for part 2
runTtest(mu1,mu2,var,150,250,"Part C - 1");
runTtest(mu1,0.2,var,150,250,"Part C - 2");

%% *Generate Data and Run T-Test*
% This function will generate two random data sets of size N1 and N2
% centered around mu1 and mu2 with a variance of var. This funciton then
% runs a ttest and plots the two data sets with gaussian fits and the two
% data sets together. It also displays the results of the ttest on the plot.
function runTtest(mu1,mu2,var,N1,N2,sub_title)
    x1 = normrnd(mu1,var,N1,1);
    x2 = normrnd(mu2,var,N2,1);
    [h,~,ci,~] = ttest2(x1,x2);
    
    %Plot subplots
    figure;
    subplot(4,4,[1 2]); histfit(x1); %X1
    title("X1 - \mu=" + num2str(mu1) + ", N=" + num2str(N1));
    subplot(4,4,[3 4]); histfit(x2); %X2
    title("X2 - \mu=" + num2str(mu2) + ", N=" + num2str(N2));
    subplot(4,4,[5 15]); histfit([x1; x2]); %X1 and X2
    title("X1 and X2");
    sgtitle(sub_title);
    
    true_mean="True Mean = " + num2str(mean([x1;x2])); %True mean of data
    %Results of null hypothesis rejection/acceptance
    if h==0
        hypothesis="   h=0; Same Mean";
    else
        hypothesis="   h=1; Different Means";
    end
    %confidence interval
    con_int = "   [" + num2str(ci(1)+(mu2-mu1)) + "," + num2str(ci(2)+(mu2-mu1)) + "]";
    
    %Text that displays results of ttest
    text = {"Conclusion","Significance Level: 5%",true_mean,...
        "Hypothesis:", hypothesis,"Confidence Interval:", con_int};
    annotation('textbox',[0.71 0 0 .5],'String',text,'FitBoxToText','on')
end