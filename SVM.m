function labels = SVM(trainingData, trainingLabels, testData )
%PCA_SVM Summary of this function goes here
%   Detailed explanation goes here

Mdl = fitcecoc(trainingData',trainingLabels);
labels = predict(Mdl,testData');



