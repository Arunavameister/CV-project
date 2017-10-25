clear all
feat=[];
num_clusters=50;
dirname = 'D:\UJM\cv project\10 images';
ext = '.jpg';
sDir=  dir( fullfile(dirname ,['*' ext]) );
length(sDir);
for d = 1:length(sDir)
    %Seq{d} = imread(sDir.name);
    images{d}=imread(fullfile('D:\UJM\cv project\10 images', sDir(d).name));
end
for d = 1:length(images)
    %Seq{d} = imread(sDir.name);
    grey_images{d}=im2single(rgb2gray(images{d}));
    [f,desc] = vl_sift(grey_images{d}) ;
    feat = [feat, desc];
end
[centers, assignments] = vl_kmeans(double(feat), num_clusters);