function mat = confusionMatrix( predLabels , trueLabels )
%CONFUSIONMATRIX Summary of this function goes here
%   Detailed explanation goes here
mat = zeros(7,7);
for i = 1:1:size(predLabels)
    mat(predLabels(i),trueLabels(i)) = mat(predLabels(i),trueLabels(i)) + 1;
end
% size(mat)
end

