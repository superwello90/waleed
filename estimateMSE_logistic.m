function [MSE] = estimateMSE_logistic(X,Y,theta,lambda)

H = 1./(1+exp(-X*theta));
MSE= mean(-Y.*log10(H)-((1-Y).*log10(1-H)))+(lambda/2).*mean(theta);
%
end