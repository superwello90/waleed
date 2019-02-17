function [THETA_normal,MSE_normal]= normal(poly,Y,lambda)

THETA_normal = (Y'*poly)/(poly'*poly);
MSE_normal = estimateMSE(poly,Y,THETA_normal',lambda);

end