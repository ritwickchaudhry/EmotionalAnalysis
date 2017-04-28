emotion1 = imread('EmotionDisplay/Angry.jpg');
emotion6 = imread('EmotionDisplay/Sad.png');
emotion5 = imread('EmotionDisplay/Happy.jpg');
emotion3 = imread('EmotionDisplay/Disgust.jpeg');
emotion4 = imread('EmotionDisplay/Fear.png');
emotion7 = imread('EmotionDisplay/Surprised.jpeg');
emotion2 = imread('EmotionDisplay/Contempt.png');

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

for i=1:numTest
    figure;
    
    if labels2(i) == 1
        emotionPic = emotion1
    elseif labels2(i) == 2
        emotionPic = emotion2
    elseif labels2(i) == 3
        emotionPic = emotion3
    elseif labels2(i) == 4
        emotionPic = emotion4
    elseif labels2(i) == 5
        emotionPic = emotion5
    elseif labels2(i) == 6
        emotionPic = emotion6
    else
        emotionPic = emotion7
    end
    
    emotionPic = imresize(emotionPic,[64 64]);   
    imshow([reshape(testData(i),[64 64]) emotionPic]);
end
    
% testModel(labels2)