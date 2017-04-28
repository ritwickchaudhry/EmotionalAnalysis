clear all;
%MODELTEST Summary of this function goes here
%   Detailed explanation goes here
dataFile = load('DataFile.mat');

totalData = dataFile.data;
split = randperm(size(totalData,2));
totalData = totalData(:,split);
totalData = totalData(:,1:1:320);
labels = dataFile.labels;
labels = labels(split);
labels = labels(1:1:320);

% unique(labels(1:200))

totalImages = 320;
numTest = 32;

% figure;
% imshow(reshape(totalData(:,1),[64 64]))
accuracies = zeros(10,1);
confMat = zeros(7,7);
for i = 1:1:10
    testImages = totalData(:,1+(i-1)*32:i*32);
    testLabels = labels(1+(i-1)*32:i*32);
    trainLabels = [labels(1:(i-1)*32) , labels(1+i*32:320)];
    trainImages = [totalData(:,1:(i-1)*32) , totalData(:,1+i*32:320)];
    [eigenVecs,meanTemplate] = generateEigenspace(trainImages);

    D_train = eigenVecs'*(trainImages -repmat(meanTemplate, [1 size(trainImages,2)]));
    %D_train = trainImages;
    %D_test = testImages;
    % unique(trainingLabels)

    D_test = eigenVecs'*(testImages - repmat(meanTemplate,[1 numTest]));
    predLabels = LDA(D_train,trainLabels , D_test);
    accuracies(i) = testModel(predLabels, testLabels);
    temp = confusionMatrix(predLabels,testLabels);
    confMat = confMat + temp;
end
meanAccuracy = mean(accuracies)



