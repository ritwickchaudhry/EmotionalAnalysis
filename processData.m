rawFile = load('DataFile');
rawData = rawFile.data;
labels = rawFile.labels;

processedData = ones(size(rawData));

for i=1:size(rawData,2)
    rawImage = rawData(:,i);
    processedData(:,i) = (rawImage - min(rawImage))/(max(rawImage) - min(rawImage));
end

data = processedData;

save('ProcessedDataFile','data','labels');