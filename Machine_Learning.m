clc
clear 
close all

excel = datastore('house_complete.csv','TreatAsMissing','NA',.....
     'MissingValue',0,'ReadSize',21613);
T = read(excel);
alpha=.0001;

for i= 1:19
    if iscell(T.(i))
       T.(i)= str2double(T.(i));
    end
end

m=floor(0.6 * length(T{:,1}));
n=floor(0.2 * length(T{:,1}));

%%%%%% training set%%%%%%%
linear_train=T{1:m,4:7};
quad_train=T{1:m,8:11};
cubic_train=T{1:m,12:15};
quartic_train=T{1:m,16:19};

Y_train=T{1:m,3};

% Normalization feature

linear_train  = normalization(linear_train);
quad_train  = normalization(quad_train);
cubic_train  = normalization(cubic_train);
quartic_train  = normalization(quartic_train);

linear_train=[ones(m,1) linear_train];
quad_train=[ones(m,1) quad_train quad_train.^2];
cubic_train=[ones(m,1) cubic_train cubic_train.^2 cubic_train.^3];
quartic_train=[ones(m,1) quartic_train quartic_train.^2 quartic_train.^3 quartic_train.^4];

Y_train = normalization(Y_train); % Normalization price

%%% cross validation set
linear_cv=T{m+1:m+n,4:7};
quad_cv=T{m+1:m+n,8:11};
cubic_cv=T{m+1:m+n,12:15};
quartic_cv=T{m+1:m+n,16:19};

Y_cv=T{m+1:m+n,3};

% Normalization features cross validation
linear_cv  = normalization(linear_cv);
quad_cv  = normalization(quad_cv);
cubic_cv  = normalization(cubic_cv);
quartic_cv  = normalization(quartic_cv);

Y_cv = normalization(Y_cv); % Normalization price

linear_cv=[ones(n,1) linear_cv];
quad_cv=[ones(n,1) quad_cv quad_cv.^2];
cubic_cv=[ones(n,1) cubic_cv cubic_cv.^2 cubic_cv.^3];
quartic_cv=[ones(n,1) quartic_cv quartic_cv.^2 quartic_cv.^3 quartic_cv.^4];


%%% test set
linear_test=T{m+n+1:end,4:7};
quad_test=T{m+n+1:end,8:11};
cubic_test=T{m+n+1:end,12:15};
quartic_test=T{m+n+1:end,16:19};

Y_test=T{m+n+1:end,3};

% Normalization features cross validation
linear_test  = normalization(linear_test);
quad_test  = normalization(quad_test);
cubic_test = normalization(cubic_test);
quartic_test  = normalization(quartic_test);
 
Y_test = normalization(Y_test); % Normalization price

linear_test=[ones(4324,1) linear_test];
quad_test=[ones(4324,1) quad_test quad_test.^2];
cubic_test=[ones(4324,1) cubic_test cubic_test.^2 cubic_test.^3];
quartic_test=[ones(4324,1) quartic_test quartic_test.^2 quartic_test.^3 quartic_test.^4];


%%%%%%% Training %%%%%%%%   

%iterations = 10000;
lambda = 2.56;

i=1;

theta_linear = zeros(size(linear_train,2),1); % Parameters (01, 02, 03, 04, 05)
theta_quad = zeros(size(quad_train,2),1); % Parameters (01, 02, ... 09)
theta_cubic = zeros(size(cubic_train,2),1); % Parameters (01, 02, ...013)
theta_quartic = zeros(size(quartic_train,2),1); % Parameters (01, 02, 03, 04, 05)

MSE_linear_train(i) = estimateMSE(linear_train,Y_train,theta_linear,lambda);
MSE_quad_train(i) = estimateMSE(quad_train,Y_train,theta_quad,lambda);
MSE_cubic_train(i) = estimateMSE(cubic_train,Y_train,theta_cubic,lambda);
MSE_quartic_train(i) = estimateMSE(quartic_train,Y_train,theta_quartic,lambda);

flag=0;
while flag==0
    
      i = i+1;

      [theta_linear,MSE_linear_train(i)]=gradient_descent_regression(linear_train,Y_train,theta_linear,alpha,m,lambda);
      [theta_quad,MSE_quad_train(i)]=gradient_descent_regression(quad_train,Y_train,theta_quad,alpha,m,lambda);
      [theta_cubic,MSE_cubic_train(i)]=gradient_descent_regression(cubic_train,Y_train,theta_cubic,alpha,m,lambda);
      [theta_quartic,MSE_quartic_train(i)]=gradient_descent_regression(quartic_train,Y_train,theta_quartic,alpha,m,lambda);
      
      if MSE_linear_train(i-1)- MSE_linear_train(i)<0
         flag=1;
      end
end

MSE_train = [ MSE_linear_train(i);MSE_quad_train(i); MSE_cubic_train(i); MSE_quartic_train(i)];
iterations = i;
%%%% Cross validation %%%%%%
i=1;

MSE_linear_cv(i) = estimateMSE(linear_cv,Y_cv,theta_linear,lambda);
MSE_quad_cv(i) = estimateMSE(quad_cv,Y_cv,theta_quad,lambda);
MSE_cubic_cv(i) = estimateMSE(cubic_cv,Y_cv,theta_cubic,lambda);
MSE_quartic_cv(i) = estimateMSE(quartic_cv,Y_cv,theta_quartic,lambda);

flag=0;
while flag==0
    
      i=i+1;

      [theta_linear,MSE_linear_cv(i)]=gradient_descent_regression(linear_cv,Y_cv,theta_linear,alpha,n,lambda);
      [theta_quad,MSE_quad_cv(i)]=gradient_descent_regression(quad_cv,Y_cv,theta_quad,alpha,n,lambda);
      [theta_cubic, MSE_cubic_cv(i)]=gradient_descent_regression(cubic_cv,Y_cv,theta_cubic,alpha,n,lambda);
      [theta_quartic,MSE_quartic_cv(i)]=gradient_descent_regression(quartic_cv,Y_cv,theta_quartic,alpha,n,lambda);
      
      if MSE_linear_cv(i-1)- MSE_linear_cv(i)<0
         flag=1;
      end
end

MSE_cv = [ MSE_linear_cv(i); MSE_quad_cv(i); MSE_cubic_cv(i); MSE_quartic_cv(i)];

if MSE_train(1)-MSE_cv(1)<0.05
   MSE_test= estimateMSE(linear_test,Y_test,theta_linear,lambda);
   %%%%% Normal Equation%%%%%%%%%%
   [theta_linear_norm,MSE_linear_norm] = normal(linear_test,Y_test,lambda);

end

if MSE_train(2)-MSE_cv(2)<0.05
   MSE_test= estimateMSE(quad_test,Y_test,theta_quad,lambda);
   %%%%% Normal Equation%%%%%%%%%%
   [theta_quad_norm,MSE_quad_norm] = normal(quad_test,Y_test,lambda);
end

if MSE_train(3)-MSE_cv(3)<0.05
   MSE_test= estimateMSE(cubic_test,Y_test,theta_cubic,lambda);
   %%%%% Normal Equation%%%%%%%%%%
   [theta_cubic_norm,MSE_cubic_norm] = normal(cubic_test,Y_test,lambda);

end

if MSE_train(4)-MSE_cv(4)<0.05
   MSE_test= estimateMSE(quartic_test,Y_test,theta_quartic,lambda);
   %%%%% Normal Equation%%%%%%%%%%
   [theta_quartic_norm,MSE_quartic_norm] = normal(quartic_test,Y_test,lambda);

end

i= 1:1:iterations;

figure('Name','MSE training set','NumberTitle','off')
plot(i,MSE_linear_train);
xlabel('no. of iterations');
ylabel('cost function');
title('linear regression') 