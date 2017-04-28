
%MODELTEST Summary of this function goes here
%   Detailed explanation goes here
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
trainLabels = labels(1:totalImages - numTest);
trainImages = totalData(:,1:totalImages-numTest);

[hog_trainingData,hog_testData] = HoGFeatures(size(trainImages,2),testImages);
D_train = [trainImages' , hog_trainingData'];
D_test = [testImages' , hog_testData'];
labels = SVM(D_train',trainLabels , D_test');
testModel(labels, testLabels)
