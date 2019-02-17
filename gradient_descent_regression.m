function [theta,MSE] = gradient_descent_regression(X,Y,theta,alpha,m,lambda)
H = X*theta;
theta = theta-alpha*(X'*(H-Y))./m;
MSE= estimateMSE(X,Y,theta,lambda);
end