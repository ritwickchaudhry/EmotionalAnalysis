function [trainingDataHOG, testDataHOG] = HoGFeatures( trainingData, testData )
%HOGFEATURES Summary of this function goes here
%   Detailed explanation goes here
numTrain = size(trainingData,2);
numTest = size(testData,2);

trainingDataHOG = ones(36*49, numTrain);

for i =1:numTrain
%     size(HoG(reshape(trainImages(:,i),[64 64])))
    trainingDataHOG(:,i) = HoG(reshape(trainingData(:,i),[64 64]));
end


testDataHOG = ones(36*49, numTest);

for i =1:numTest
    testDataHOG(:,i) = HoG(reshape(testData(:,i),[64 64]));
end

end

