warning('off','all');
warning;
%% whole data is unusable beacuse of small ram 64GB
train_idx = randperm (10000);
test_idx = randperm (3000);
myVars = {'train_nnx','train_nny','val_x','val_y','test_x','test_y'};

%% 351 feature
load('data/regression_351.mat',myVars{:});
X = train_nnx(train_idx, :);
Y = train_nny(train_idx, :);
X_test = test_x(test_idx, :);
Y_test = test_y(test_idx, :);
%% Twin Gaussian Processes parameters
% Param.kparam1 = 2*1e-4;
% Param.kparam2 = 2*1e-6;
% Param.lambda = 1e-3;

Param.kparam1 = 2*1e-4;
Param.kparam2 = 6.08;
Param.lambda = 1e-3;
%% feature transform through NN
load('temp/A1.mat');
nnT_X = nnTransform(nn_A1, X);
nnT_X_test = nnTransform(nn_A1, X_test);
%% Twin Gaussian Processes with NN transformed features
[InvIK, InvOK] = TGPTrain(nnT_X, Y, Param);
predicted_f0 = TGPTest(nnT_X_test, nnT_X, Y, Param, InvIK, InvOK);
[Error_A1, TGPErrorvec] = JointError(predicted_f0, Y_test);
disp(['TGP: ' num2str(Error_A1)]);

%% feature transform through NN
load('temp/B1.mat');
nnT_X = nnTransform(nn_B1, X);
nnT_X_test = nnTransform(nn_B1, X_test);
%% Twin Gaussian Processes with NN transformed features
[InvIK, InvOK] = TGPTrain(nnT_X, Y, Param);
predicted_f0 = TGPTest(nnT_X_test, nnT_X, Y, Param, InvIK, InvOK);
[Error_B1, TGPErrorvec] = JointError(predicted_f0, Y_test);
disp(['TGP: ' num2str(Error_B1)]);

%% feature transform through NN
load('temp/C1.mat');
nnT_X = nnTransform(nn_C1, X);
nnT_X_test = nnTransform(nn_C1, X_test);
%% Twin Gaussian Processes with NN transformed features
[InvIK, InvOK] = TGPTrain(nnT_X, Y, Param);
predicted_f0 = TGPTest(nnT_X_test, nnT_X, Y, Param, InvIK, InvOK);
[Error_C1, TGPErrorvec] = JointError(predicted_f0, Y_test);
disp(['TGP: ' num2str(Error_C1)]);

%% feature transform through NN
load('temp/D1.mat');
nnT_X = nnTransform(nn_D1, X);
nnT_X_test = nnTransform(nn_D1, X_test);
%% Twin Gaussian Processes with NN transformed features
[InvIK, InvOK] = TGPTrain(nnT_X, Y, Param);
predicted_f0 = TGPTest(nnT_X_test, nnT_X, Y, Param, InvIK, InvOK);
[Error_D1, TGPErrorvec] = JointError(predicted_f0, Y_test);
disp(['TGP: ' num2str(Error_D1)]);

save('temp/error_TGP.mat','Error_A1','Error_B1','Error_C1','Error_D1');
