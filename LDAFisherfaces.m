clear all;
% Dummy Data
% totalData = ones(10,100);

dataFile = load('DataFile.mat');

totalData = dataFile.data;
split = randperm(size(totalData,2));
totalData = totalData(:,split);
labels = dataFile.labels;
labels = labels(split);

% unique(labels(1:200))

totalImages = 322;
numTest = 30;

% figure;
% imshow(reshape(totalData(:,1),[64 64]))

testImages = totalData(:,totalImages - numTest + 1:totalImages);
testLabels = labels(totalImages - numTest + 1:totalImages);

trainImages = totalData(:,1:totalImages-numTest);

[eigenVecs,meanTemplate] = generateEigenspace(trainImages);

trainingData = eigenVecs'*(trainImages -repmat(meanTemplate, [1 size(trainImages,2)]));
trainingLabels = labels(1,1:totalImages-numTest);

% unique(trainingLabels)

testData = eigenVecs'*(testImages - repmat(meanTemplate,[1 numTest]));

% labels = LDA(trainingData, trainingLabels, testData);
% testModel(labels,testLabels)

labels = LDA_HOG_SVM(trainingData, trainingLabels, testData, trainImages, testImages);
testModel(labels,testLabels)
