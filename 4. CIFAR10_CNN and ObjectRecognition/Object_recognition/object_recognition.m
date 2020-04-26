clear all;
imds = imageDatastore('101_ObjectCategories', 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

%Balance the dataset
imds = balance_dataset(imds);

%Splitting the data
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');

%loading the network
% net = alexnet;
% net = resnet18;
% net = mobilenetv2;
% net = vgg19;
net = resnet101;
inputSize = net.Layers(1).InputSize;

%Resizing the data
augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augimdsTest = augmentedImageDatastore(inputSize,imdsTest, 'ColorPreprocessing', 'gray2rgb');

%Defining the feature layer
% featureLayer = 'fc7'; %for alexnet
% featureLayer = 'fc1000'; %for resnet
% featureLayer = 'Logits'; %for mobilenet
% featureLayer = 'fc7'; %for vgg19
featureLayer = 'res5c_relu'; %for resnet101

trainingFeatures = activations(net, augimdsTrain, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
trainingLabels = imdsTrain.Labels;

% Train multiclass SVM classifier using a fast linear solver

classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');




% Extract test features using the CNN
testFeatures = activations(net, augimdsTest, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Get the known labels
testLabels = imdsTest.Labels;

%compute accuracy
accuracy = mean(predictedLabels == testLabels)


%A function to balance the dataset
function imds = balance_dataset(imds)
    tbl = countEachLabel(imds);
    % Determine the smallest amount of images in a category
    minSetCount = min(tbl{:,2}); 

    % Limit the number of images 
    maxNumImages = 100;
    minSetCount = min(maxNumImages,minSetCount);

    % trim the dataset.
    imds = splitEachLabel(imds, minSetCount, 'randomize');
end