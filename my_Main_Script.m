% Choose directory containing the data
dir_data = uigetdir();

files_emotion = dir([dir_data,'\Emotion_Labels\Emotion\']);
formatSpec = '%f';

% Face Detector object for face extraction in each image
faceDetector = vision.CascadeObjectDetector;

% Feature Vectors of different images: Each column represents a HoG Feature
% vector
feature_vector = zeros(36*49,1000);
Y = zeros(1,1000);
count = 0;

for i = 3:size(files_emotion)
    % Looking for emotion labels in each person's data
    disp(['Extracting from ',files_emotion(i).name]);
    s_i = dir([dir_data,'\Emotion_Labels\Emotion\',files_emotion(i).name,'\']);
    
    for j = 3:size(s_i)    
        % Looking for different emotion faces in each person's images
        disp(['      ',s_i(j).name]);
        s_i_j = dir([dir_data,'\Emotion_Labels\Emotion\',files_emotion(i).name,'\',s_i(j).name]);
        
        if size(s_i_j,1) > 2
            % If emotion label exists, extract emotion label from image
            textfile_emotion = fopen([dir_data,'\Emotion_Labels\Emotion\',files_emotion(i).name,'\',s_i(j).name,'\',s_i_j(3).name],'r');
            labels = int32(fscanf(textfile_emotion,formatSpec));
            fclose(textfile_emotion);
             
            % Extract peak image for data
            image_dir = dir([dir_data,'\extended-cohn-kanade-images\cohn-kanade-images\',files_emotion(i).name,'\',s_i(j).name]);
            peak_image_name = [dir_data,'\extended-cohn-kanade-images\cohn-kanade-images\',files_emotion(i).name,'\',s_i(j).name,'\',image_dir(size(image_dir,1)).name];
            I = imread(peak_image_name);
            
            
            % Image I contains the data
            % labels contains the coressponding emotion
            
            
            %Extract bounding box of face from Image
            bboxes = step(faceDetector,I);
            if bboxes(1,3) > 50
                count = count + 1;
                feature_vector(:,count) = HoG(I(bboxes(1,2):bboxes(1,2)+bboxes(1,4),bboxes(1,1):bboxes(1,1)+bboxes(1,3)));
                Y(1,count) = labels;
%                 figure(1);
%                 imshow(I(bboxes(1,2):bboxes(1,2)+bboxes(1,4),bboxes(1,1):bboxes(1,1)+bboxes(1,3)));
            end
        end
    end
    
    
end

min(Y)
max(Y)
feature_vector = feature_vector(:,1:count);
numEmotions = 8;
N = count;