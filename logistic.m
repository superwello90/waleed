clc
clear 
close all

excel = datastore('heart_DD.csv','TreatAsMissing','NA',.....
     'MissingValue',0,'ReadSize',251);
T = read(excel);
alpha=.001;

for i= 1:14
    if iscell(T.(i))
       T.(i)= str2double(T.(i));
    end
end

m=floor(0.6 * length(T{:,1}));
n=floor(0.2 * length(T{:,1}));

%%%%%% training set%%%%%%%

linear_train=T{1:m,1:3};
quad_train=T{1:m,4:6};
cubic_train=T{1:m,7:9};
quartic_train=T{1:m,10:13};

Y_train=T{1:m,14};

% Normalization feature
linear_train  = normalization(linear_train);
quad_train  = normalization(quad_train);
cubic_train  = normalization(cubic_train);
quartic_train  = normalization(quartic_train);

linear_train=[ones(m,1) linear_train];
quad_train=[ones(m,1) quad_train quad_train.^2];
cubic_train=[ones(m,1) cubic_train cubic_train.^2 cubic_train.^3];
quartic_train=[ones(m,1) quartic_train quartic_train.^2 quartic_train.^3 quartic_train.^4];

lambda = 2.56;

i=1;

theta_linear = zeros(size(linear_train,2),1); % Parameters (01, 02, 03, 04, 05)
theta_quad = zeros(size(quad_train,2),1); % Parameters (01, 02, ... 09)
theta_cubic = zeros(size(cubic_train,2),1); % Parameters (01, 02, ...013)
theta_quartic = zeros(size(quartic_train,2),1); % Parameters (01, 02, 03, 04, 05)
H = 1./(1+exp(-linear_train*theta_linear));

MSE= mean(-Y_train.*log10(H))+ mean(-(1-Y_train).*log10(1-H));

MSE_linear_train(i) = estimateMSE_logistic(linear_train,Y_train,theta_linear,lambda);
MSE_quad_train(i) = estimateMSE_logistic(quad_train,Y_train,theta_quad,lambda);
MSE_cubic_train(i) = estimateMSE_logistic(cubic_train,Y_train,theta_cubic,lambda);
MSE_quartic_train(i) = estimateMSE_logistic(quartic_train,Y_train,theta_quartic,lambda);

flag=0;
while flag==0
    
      i = i+1;
   
      [theta_linear,MSE_linear_train(i)]=gradient_descent_logistic(linear_train,Y_train,theta_linear,alpha,m,lambda);
      [theta_quad,MSE_quad_train(i)]=gradient_descent_logistic(quad_train,Y_train,theta_quad,alpha,m,lambda);
      [theta_cubic,MSE_cubic_train(i)]=gradient_descent_logistic(cubic_train,Y_train,theta_cubic,alpha,m,lambda);
      [theta_quartic,MSE_quartic_train(i)]=gradient_descent_logistic(quartic_train,Y_train,theta_quartic,alpha,m,lambda);
      
      if MSE_linear_train(i-1)- MSE_linear_train(i)<0 || i==5000
         flag=1;
      end
end

MSE_train = [ MSE_linear_train(i);MSE_quad_train(i); MSE_cubic_train(i); MSE_quartic_train(i)];

k= 1:1:i;

figure('Name','training logistic regression','NumberTitle','off')
plot(k,MSE_linear_train);
hold on;
xlabel('no. of iterations');
ylabel('cost function');
title('linear polynomial ') 

figure('Name','training logistic regression','NumberTitle','off')
plot(k,MSE_quad_train);
hold on;
xlabel('no. of iterations');
ylabel('cost function');
title('quad polynomial ') 

%%%% Cross validation %%%%%%
linear_cv=T{m+1:m+n,1:3};
quad_cv=T{m+1:m+n,4:6};
cubic_cv=T{m+1:m+n,7:9};
quartic_cv=T{m+1:m+n,10:13};

Y_cv=T{m+1:m+n,3};

% Normalization features cross validation
linear_cv  = normalization(linear_cv);
quad_cv  = normalization(quad_cv);
cubic_cv  = normalization(cubic_cv);
quartic_cv  = normalization(quartic_cv);

linear_cv=[ones(n,1) linear_cv];
quad_cv=[ones(n,1) quad_cv quad_cv.^2];
cubic_cv=[ones(n,1) cubic_cv cubic_cv.^2 cubic_cv.^3];
quartic_cv=[ones(n,1) quartic_cv quartic_cv.^2 quartic_cv.^3 quartic_cv.^4];

i=1;

MSE_linear_cv(i) = estimateMSE_logistic(linear_cv,Y_cv,theta_linear,lambda);
MSE_quad_cv(i) = estimateMSE_logistic(quad_cv,Y_cv,theta_quad,lambda);
MSE_cubic_cv(i) = estimateMSE_logistic(cubic_cv,Y_cv,theta_cubic,lambda);
MSE_quartic_cv(i) = estimateMSE_logistic(quartic_cv,Y_cv,theta_quartic,lambda);

flag=0;
while flag==0
    
      i=i+1;

      [theta_linear,MSE_linear_cv(i)]=gradient_descent_logistic(linear_cv,Y_cv,theta_linear,alpha,n,lambda);
      [theta_quad,MSE_quad_cv(i)]=gradient_descent_logistic(quad_cv,Y_cv,theta_quad,alpha,n,lambda);
      [theta_cubic, MSE_cubic_cv(i)]=gradient_descent_logistic(cubic_cv,Y_cv,theta_cubic,alpha,n,lambda);
      [theta_quartic,MSE_quartic_cv(i)]=gradient_descent_logistic(quartic_cv,Y_cv,theta_quartic,alpha,n,lambda);
      
      if MSE_linear_cv(i-1)- MSE_linear_cv(i)<0 || i == 5000
         flag=1;
      end
end

MSE_cv = [ MSE_linear_cv(i); MSE_quad_cv(i); MSE_cubic_cv(i); MSE_quartic_cv(i)];

k= 1:1:i;

figure('Name','logistic cross validation MSE','NumberTitle','off')
plot(k,MSE_linear_cv);
hold off;
xlabel('no. of iterations');
ylabel('cost function');
title('linear polynomial') 

figure('Name','logistic cross validation MSE','NumberTitle','off')
plot(k,MSE_quad_cv);
hold off;
xlabel('no. of iterations');
ylabel('cost function');
title('quadratic polynimoial') 

%%% test set
linear_test=T{m+n+1:end,1:3};
quad_test=T{m+n+1:end,4:6};
cubic_test=T{m+n+1:end,7:9};
quartic_test=T{m+n+1:end,10:13};

Y_test=T{m+n+1:end,3};

% Normalization features test
linear_test  = normalization(linear_test);
quad_test  = normalization(quad_test);
cubic_test = normalization(cubic_test);
quartic_test  = normalization(quartic_test);
 
Y_test = normalization(Y_test); % Normalization price

linear_test=[ones(n,1) linear_test];
quad_test=[ones(n,1) quad_test quad_test.^2];
cubic_test=[ones(n,1) cubic_test cubic_test.^2 cubic_test.^3];
quartic_test=[ones(n,1) quartic_test quartic_test.^2 quartic_test.^3 quartic_test.^4];

if MSE_train(1)-MSE_cv(1)<0.05
   MSE_test= estimateMSE_logistic(linear_test,Y_test,theta_linear,lambda);
   %%%%% Normal Equation%%%%%%%%%%
   [theta_linear_norm,MSE_linear_norm] = normal(linear_test,Y_test,lambda);

end

if MSE_train(2)-MSE_cv(2)<0.05
   MSE_test= estimateMSE_logistic(quad_test,Y_test,theta_quad,lambda);
   %%%%% Normal Equation%%%%%%%%%%
   [theta_quad_norm,MSE_quad_norm] = normal(quad_test,Y_test,lambda);
end

if MSE_train(3)-MSE_cv(3)<0.05
   MSE_test= estimateMSE_logistic(cubic_test,Y_test,theta_cubic,lambda);
   %%%%% Normal Equation%%%%%%%%%%
   [theta_cubic_norm,MSE_cubic_norm] = normal(cubic_test,Y_test,lambda);

end

if MSE_train(4)-MSE_cv(4)<0.05
   MSE_test= estimateMSE_logistic(quartic_test,Y_test,theta_quartic,lambda);
   %%%%% Normal Equation%%%%%%%%%%
   [theta_quartic_norm,MSE_quartic_norm] = normal(quartic_test,Y_test,lambda);

end

% % Normalization feature
%  for k = 1:size(train_linear,2)
%      train_linear(:,k)  = (train_linear(:,k)-mean(train_linear(:,k)))/std(train_linear(:,k));
%      train_quad(:,k)  = (train_quad(:,k)-mean(train_quad(:,k)))/std(train_quad(:,k));
%      train_cubic(:,k)  = (train_cubic(:,k)-mean(train_cubic(:,k)))/std(train_cubic(:,k));
%      train_quartic(:,k)  = (train_quartic(:,k)-mean(train_quartic(:,k)))/std(train_quartic(:,k));
%  end
