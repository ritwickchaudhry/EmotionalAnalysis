function accuracy = LDA(trainingData, trainingLabels, testData, testLabels)

numLabels = 7;
epsilon = 0;
[imageVecSize,dataSize] = size(trainingData);
numData = ones(1,numLabels);

% numData = [10,10,20,20,10,20,10];

% Find the Means of all the classes and the Covariance Matrices
means = ones(imageVecSize,numLabels);
covarianceMatrices = ones(imageVecSize,imageVecSize,numLabels);

classData = cell(1,7);

for i=1:numLabels
    indices = find(trainingLabels == i);
%     classData = trainingData(:,cumulativeCount+1:cumulativeCount + numData(i));
    classData{i} = trainingData(:,indices);
    numData(i) = size(classData{i},2);
    means(:,i) = mean(classData{i},2);
%     cumulativeCount = cumulativeCount + numData(i);
%     size(classData{i})
%     numData(i)
    meanSubData = classData{i} - repmat(means(:,i),[1 numData(i)]);
    covarianceMatrices(:,:,i) = meanSubData*meanSubData';
%     covarianceMatrices(:,:,i) = cov(trainingData(:,indices)');
end

% numData

totalMean = mean(trainingData,2);

% size(covarianceMatrices(:,:,1))

% Within Class Scatter
Sw = sum(covarianceMatrices,3);

% Scatter Between Classes
Sb = ones(imageVecSize,imageVecSize);
for i=1:numLabels
    meanDiff  = means(:,i) - totalMean;
    Sb = Sb + numData(i) * (meanDiff*meanDiff');
end
det(Sb)

% Adding the Regulariser Term
Sw = Sw + epsilon*eye(size(Sw,1));
% rank(Sw)

[V,D] = eigs(inv(Sw)*Sb);
Weights = V(:,1:numLabels-1);

% size(means)
% size(Weights)

meanProjections = Weights' * means;
numTest = size(testData,2);
accuracy = 0;
for j=1:numTest
    testProj = Weights' * testData(:,j);
    minDist = inf;
    predLabel = 0;
    for i=1:numLabels
        if(norm(testProj - meanProjections(i)) < minDist)
            minDist = norm(testProj - meanProjections(i));
            predLabel = i;
        end
    end
    
    if predLabel == testLabels(j)
        accuracy = accuracy + 1;
    end
end
accuracy = (accuracy*1.0/numTest)*100
