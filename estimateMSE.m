function [MSE] = estimateMSE(X,Y,theta,lambda)

H = X*theta;
MSE= mean((H-Y).^2)/2+(lambda/2).*mean(theta);

end