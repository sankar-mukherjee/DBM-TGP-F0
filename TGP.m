warning('off','all');
warning;
addpath('TGP');

load('weights/dbn_NN_Class269_200_200_200.mat');
%% whole data is unusable beacuse of small ram 64GB
train_idx = randperm (10000);
test_idx = randperm (3000);
myVars = {'train_nnx','train_nny','val_x','val_y','test_x','test_y'};
load('data/regression_269.mat',myVars{:});
X = train_nnx(train_idx, :);
Y = train_nny(train_idx, :);
X_test = test_x(test_idx, :);
Y_test = test_y(test_idx, :);
%% feature transform through NN
nn = dbnunfoldtonn(dbn, 327);       % 327 is meannig less only dbn --> nn structure
nnT_X = nnTransform(nn, X);
nnT_X_test = nnTransform(nn, X_test);
%% Twin Gaussian Processes
Param.kparam1 = 0.2;Param.kparam2 = 2*1e-6;Param.kparam3 = Param.kparam2;Param.lambda = 1e-3;Param.knn = 100;
% with NN transformed features
[InvIK, InvOK] = TGPTrain(nnT_X, Y, Param);
predicted_f0 = TGPTest(nnT_X_test, nnT_X, Y, Param, InvIK, InvOK);
[Error, TGPErrorvec] = JointError(predicted_f0, Y_test);
disp(['TGP: ' num2str(Error)]);
save('weights/TGP_10000.mat', 'InvIK','InvOK','train_idx','test_idx','Param');


nnT_X = nnTransform(nn_f, X);
nnT_X_test = nnTransform(nn_f, X_test);
%% Twin Gaussian Processes
Param.kparam1 = 0.2;Param.kparam2 = 2*1e-6;Param.kparam3 = Param.kparam2;Param.lambda = 1e-3;Param.knn = 100;
% with NN transformed features
[InvIK, InvOK] = TGPTrain(nnT_X, Y, Param);
predicted_f0 = TGPTest(nnT_X_test, nnT_X, Y, Param, InvIK, InvOK);
[Error, TGPErrorvec] = JointError(predicted_f0, Y_test);
disp(['TGP: ' num2str(Error)]);
save('weights/TGP_10000_f.mat', 'InvIK','InvOK','train_idx','test_idx','Param');


nnT_X = nnTransform(nn_d, X);
nnT_X_test = nnTransform(nn_d, X_test);
%% Twin Gaussian Processes
Param.kparam1 = 0.2;Param.kparam2 = 2*1e-6;Param.kparam3 = Param.kparam2;Param.lambda = 1e-3;Param.knn = 100;
% with NN transformed features
[InvIK, InvOK] = TGPTrain(nnT_X, Y, Param);
predicted_f0 = TGPTest(nnT_X_test, nnT_X, Y, Param, InvIK, InvOK);
[Error, TGPErrorvec] = JointError(predicted_f0, Y_test);
disp(['TGP: ' num2str(Error)]);
save('weights/TGP_10000_d.mat', 'InvIK','InvOK','train_idx','test_idx','Param');


nnT_X = nnTransform(nn_dd, X);
nnT_X_test = nnTransform(nn_dd, X_test);
%% Twin Gaussian Processes
Param.kparam1 = 0.2;Param.kparam2 = 2*1e-6;Param.kparam3 = Param.kparam2;Param.lambda = 1e-3;Param.knn = 100;
% with NN transformed features
[InvIK, InvOK] = TGPTrain(nnT_X, Y, Param);
predicted_f0 = TGPTest(nnT_X_test, nnT_X, Y, Param, InvIK, InvOK);
[Error, TGPErrorvec] = JointError(predicted_f0, Y_test);
disp(['TGP: ' num2str(Error)]);
save('weights/TGP_10000_dd.mat', 'InvIK','InvOK','train_idx','test_idx','Param');