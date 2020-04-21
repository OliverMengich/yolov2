
%% 
load('CarData.mat');
imageDatatrain = imageDatastore(CarData{:,'imageFilename'});
boxDataTrain = boxLabelDatastore(CarData(:,'car'));
%%
trainingData = combine(imageDatatrain,boxDataTrain);
data = read(trainingData);
I = data{1};
bbox = data{2};
imshow(insertShape(I,'rectangle',bbox));
%% Specify the image imput size and number of object to be detected
inputSize = [224 224 3];

numClasses = width(CarData)-1;
%% Estimate the anchor boxes
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

%% Load the pretrained network
featureExtractionNetwork = resnet50;

featureLayer = 'activation_40_relu';

lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);
%%
augmentedTrainingData = transform(trainingData,@augmentData);
%%
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)

%%

preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));

data = read(preprocessedTrainingData);

I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
%%

options = trainingOptions('sgdm', ...
        'MiniBatchSize', 16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',25,...
        'CheckpointPath', tempdir, ...
        'Shuffle','never',...
        'Verbose',true);
    
 [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
 %%
 image = imread('carsgraz_145.jpg');
%imshow(image)
image1 = imresize(image,[224 224]);
bbox = detect(detector,image1);
imshow(insertObjectAnnotation(image1,'rectangle',(bbox),'Car'))
 %%
 videoread= vision.VideoFileReader('bikeTorr.mp4'); 
 writer = vision.VideoFileWriter('result.avi');
 player = vision.DeployableVideoPlayer;
 while ~isDone(videoread)
     frame = step(videoread);
     frame = imresize(frame,[224 224]);
     bbox = detect(detector,frame);
     J = insertShape(frame,'rectangle',bbox);
     step(writer,J);
 end
%%
function B = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.
B = cell(size(A));

I = A{1};
sz = size(I);
if numel(sz)==3 && sz(3) == 3
    I = jitterColorHSV(I,...
        'Contrast',0.2,...
        'Hue',0,...
        'Saturation',0.1,...
        'Brightness',0.2);
end

% Randomly flip and scale image.
tform = randomAffine2d('XReflection',true,'Scale',[1 1.1]);
rout = affineOutputView(sz,tform,'BoundsStyle','CenterOutput');
B{1} = imwarp(I,tform,'OutputView',rout);

% Apply same transform to boxes.
[B{2},indices] = bboxwarp(A{2},tform,rout,'OverlapThreshold',0.25);
B{3} = A{3}(indices);

% Return original data only when all boxes are removed by warping.
if isempty(indices)
    B = A;
end
end

function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to the targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
data{2} = bboxresize(data{2},scale);
end