function accuracy = testModel(labels , testLabels )
%TESTMODEL Summary of this function goes here
%   Detailed explanation goes here
c =  0;
for i = 1:1:size(labels,1)
    if(labels(i) == testLabels(i))
        c = c+1;
    end    
end
accuracy = (c*1.0/size(labels,1))*100.0;
end


