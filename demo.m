test_dir = uigetdir();
files = dir(test_dir);
testData = zeros(4096,size(files,1)-2);
count = 1;
for file = files'
    if(file.name(1,1) ~= '.')
        file.name
        testData(:,count) = face_detect(strcat(test_dir,'\',file.name));
        count = count + 1;
    end
end

dataFile = load('ProcessedDataFile.mat');

trainingData = dataFile.data;
trainingLabels = dataFile.labels;
numTest = size(testData,2);
[eigenVecs,meanTemplate] = generateEigenspace(trainingData);
SVM(trainingData, trainingLabels, testData)

trainingData = eigenVecs'*(trainingData -repmat(meanTemplate, [1 size(trainingData,2)]));


testData = eigenVecs'*(testData - repmat(meanTemplate,[1 numTest]));

%LDA(trainingData, trainingLabels, testData)
%SVM(trainingData, trainingLabels, testData)
%LDA_SVM(trainingData, trainingLabels, testData)