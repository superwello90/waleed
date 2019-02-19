function [theta,MSE] = gradient_descent_logistic(X,Y,theta,alpha,m,lambda)
H = 1./(1+exp(-X*theta));
theta = theta-alpha*(X'*(H-Y))./m;
MSE= estimateMSE(X,Y,theta,lambda);
end