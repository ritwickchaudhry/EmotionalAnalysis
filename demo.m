test_dir = '..\demo';
% test_dir = uigetdir();
files = dir(test_dir);
testData = zeros(4096,size(files,1)-2);
count = 1;
for file = files'
    if(file.name(1,1) ~= '.')
        file.name
        testData(:,count) = face_detect(strcat(test_dir,'\',file.name));
%         figure;
%         imshow(reshape(testData(:,count),[64 64]),[]);
        count = count + 1;
    end
end


dataFile = load('DataFile.mat');
totalData = dataFile.data;

% figure;
% imshow(reshape(totalData(:,1),[64 64]), []);

% testData(:,2) - totalData(:,1)

split = randperm(size(totalData,2));
totalData = totalData(:,split);
labels = dataFile.labels;
labels = labels(split);

trainingData = totalData;
trainingLabels = labels;

numTest = size(testData,2);
[eigenVecs,meanTemplate] = generateEigenspace(trainingData);
% numTest = size(totalData,2);
% testData = trainingData;
%labels1 =  SVM(trainingData, trainingLabels, testData)
trainingData2 = trainingData;
testData2 = testData;
trainingData = eigenVecs'*(trainingData -repmat(meanTemplate, [1 size(trainingData,2)]));


testData = eigenVecs'*(testData - repmat(meanTemplate,[1 numTest]));

%LDA(trainingData, trainingLabels, testData)
%SVM(trainingData, trainingLabels, testData)
labels2 = LDA_HOG_SVM(trainingData, trainingLabels, testData, trainingData2, testData2)
% testModel(labels2)