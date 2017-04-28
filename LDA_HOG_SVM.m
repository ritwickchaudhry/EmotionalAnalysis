function labels = LDA_HOG_SVM(trainingDataProjected, trainingLabels, testDataProjected, trainingData, testData)
numLabels = 7;
epsilon = 0;
[imageVecSize,~] = size(trainingDataProjected);
numData = ones(1,numLabels);

% numData = [10,10,20,20,10,20,10];

% Find the Means of all the classes and the Covariance Matrices
means = ones(imageVecSize,numLabels);
covarianceMatrices = ones(imageVecSize,imageVecSize,numLabels);

classData = cell(1,7);
for i=1:numLabels
    indices = find(trainingLabels == i);
%     classData = trainingData(:,cumulativeCount+1:cumulativeCount + numData(i));
    classData{i} = trainingDataProjected(:,indices);
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

totalMean = mean(trainingDataProjected,2);

% size(covarianceMatrices(:,:,1))

% Within Class Scatter
Sw = sum(covarianceMatrices,3);
% max(max(Sw))

% Scatter Between Classes
Sb = ones(imageVecSize,imageVecSize);
for i=1:numLabels
    meanDiff  = means(:,i) - totalMean;
    Sb = Sb + numData(i) * (meanDiff*meanDiff');
end
% save('SbFile','Sb');
% det(Sb)
% Adding the Regulariser Term
Sw = Sw + epsilon*eye(size(Sw,1));

% Reduce the Rank of Sw
% [U,S,V] = svd(Sw);
% sz = size(S,2);
% for i=1:10
%     S(sz-(i-1),sz-(i-1)) = 0;
% end
% Sw = U*S*V';

invSw = inv(Sw);
% save('invSwFile2','invSw');

% Take the Top EigenVectors
[V,D] = eig(invSw*Sb);
[B,I] = sort(diag(D),'descend');
% size(I)
I = I(1:numLabels-1,1);
V = V(:,I);
Weights = V;

% Weights = V(:,1:numLabels-1);
[hog_trainingData,hog_testData] = HoGFeatures(trainingData,testData);

D_train =  [trainingDataProjected' * Weights ,hog_trainingData'];
L_train = trainingLabels;
D_test = [testDataProjected' * Weights , hog_testData'];
Mdl = fitcecoc(D_train,L_train);
labels = predict(Mdl,D_test);

