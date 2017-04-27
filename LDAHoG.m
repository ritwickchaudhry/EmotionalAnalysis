clear all;
% Dummy Data
% totalData = ones(10,100);

dataFile = load('DataFile.mat');

totalData = dataFile.data;
labels = dataFile.labels;

totalImages = 322;
numTest = 20;
numTrain = totalImages - numTest;

testImages = totalData(:,totalImages - numTest + 1:totalImages);
testLabels = labels(totalImages - numTest + 1:totalImages);

trainImages = totalData(:,1:totalImages-numTest);

trainingData = ones(36*49, numTrain);

for i =1:numTrain
%     size(HoG(reshape(trainImages(:,i),[64 64])))
    trainingData(:,i) = HoG(reshape(trainImages(:,i),[64 64]));
end
trainingLabels = labels(1,1:totalImages-numTest);


testData = ones(36*49, numTest);

for i =1:numTest
    testData(:,i) = HoG(reshape(testImages(:,i),[64 64]));
end

LDA(trainingData,trainingLabels,testData,testLabels);