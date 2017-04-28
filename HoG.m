function [ HoG ] = HoG( I )
im1 = imresize(I,[64 64]);



im1_x = zeros(64,64);
im1_y = zeros(64,64);

for i = 1:64
    for j = 1:64        
        im1_x(i,j) = double(im1(min(i+1,64),j))-double(im1(max(1,i-1),j));
        im1_y(i,j) = double(im1(i,min(j+1,64)))-double(im1(i,max(j-1,1)));
    end
end
im_dir =  double(mod(atan2(im1_y,im1_x).*180/pi,180));

im_mag = double((im1_x.^2 + im1_y.^2).^0.5);


% Patch based Collection of Histograms
Histogram_Patch = zeros(9,64);
Norm_Histogram_Patch = zeros(36,49);
for i = 1:8
    for j = 1:8
        for k = 1:8
            for l = 1:8
                mag = im_mag((i-1)*8+ k, (j-1)*8 +l);
                dir = im_dir((i-1)*8+ k, (j-1)*8 +l);
                dir_index1 = floor(dir/20)+1;
                dir_index2 = ceil(dir/20)+1;
                if(dir_index2 == 10)
                    dir_index2 = 1;
                end

                ratio = mod(dir,20)/20.0;
                
                if mod(dir,20) ~= 0
                    Histogram_Patch(dir_index1, (i-1)*8 + j) = Histogram_Patch(dir_index1, (i-1)*8 + j) + (mag*ratio);
                    Histogram_Patch(dir_index2, (i-1)*8 + j) = Histogram_Patch(dir_index2, (i-1)*8 + j) + (mag*(1-ratio));
                end
                if mod(dir,20) == 0
                    Histogram_Patch(int32(dir/20)+1,(i-1)*8+j) = Histogram_Patch(int32(dir/20)+1,(i-1)*8+j) + mag;
                end
                    
            end
        end
    end
end
% Patch Normalisation
for i = 1:7
    for j = 1:7
        
        Norm_Histogram_Patch(1:9,(i-1)*7+j) = Histogram_Patch(:,(i-1)*8+j);
        Norm_Histogram_Patch(10:18,(i-1)*7+j) = Histogram_Patch(:,(i-1)*8+j+1);
        Norm_Histogram_Patch(19:27,(i-1)*7+j) = Histogram_Patch(:,i*8+j);
        Norm_Histogram_Patch(28:36,(i-1)*7+j) = Histogram_Patch(:,i*8+j+1);
        if norm(Norm_Histogram_Patch(:,(i-1)*7+j)) ~= 0 
            Norm_Histogram_Patch(:,(i-1)*7+j) = Norm_Histogram_Patch(:,(i-1)*7+j) / (norm(Norm_Histogram_Patch(:,(i-1)*7+j)));
        end
    end
    
end

HoG = Norm_Histogram_Patch(:);
end

