function face = face_detect( filename )
%FACE_DETECT Summary of this function goes here
%   Detailed explanation goes here
faceDetector = vision.CascadeObjectDetector;
I = imread(filename);
if(size(I,3)==3)
    I = rgb2gray(I);
end
% Image I contains the data
% labels contains the coressponding emotion

%Extract bounding box of face from Image
bboxes = step(faceDetector,I);
face = I(bboxes(1,2):bboxes(1,2)+bboxes(1,4),bboxes(1,1):bboxes(1,1)+bboxes(1,3));
face = imresize(face,[64 64]);
face = face(:);
% face = (face - min(face))/(max(face) - min(face));
