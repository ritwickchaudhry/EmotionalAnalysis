function [eigVecs,meanTemplate] = generateEigenSpace(data)

meanTemplate = mean(data,2);
data = data - repmat(meanTemplate,[1 size(data,2)]);
[U,S,V] = svd(data, 'econ');
% size(V)
V = V(:,1:64);
eigVecs = data*V;
eigVecs = normc(eigVecs);
% size(eigVecs)