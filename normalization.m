function [feature] = normalization(feature)

for i=1:size(feature,2)
feature(:,i)  = (feature(:,i)-mean(feature(:,i)))/std(feature(:,i));
end

end