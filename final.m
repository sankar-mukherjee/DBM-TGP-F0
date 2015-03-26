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
Param.kparam1 = 2*1e-5;
Param.kparam2 = 6.08;
Param.lambda = 1e-3;

%% feature transform through NN
load('temp/A1.mat');
nnT_X = nnTransform(nn_A1, X);
nnT_X_test = nnTransform(nn_A1, X_test);
%% Twin Gaussian Processes with NN transformed features
[InvIK, InvOK] = TGPTrain(nnT_X, Y, Param);
predicted_f0_A = TGPTest(nnT_X_test, nnT_X, Y, Param, InvIK, InvOK);
[Error_A1, TGPErrorvec] = JointError(predicted_f0_A, Y_test);
disp(['TGP: ' num2str(Error_A1)]);

%% feature transform through NN
load('temp/B1.mat');
nnT_X = nnTransform(nn_B1, X);
nnT_X_test = nnTransform(nn_B1, X_test);
%% Twin Gaussian Processes with NN transformed features
[InvIK, InvOK] = TGPTrain(nnT_X, Y, Param);
predicted_f0_B = TGPTest(nnT_X_test, nnT_X, Y, Param, InvIK, InvOK);
[Error_B1, TGPErrorvec] = JointError(predicted_f0_B, Y_test);
disp(['TGP: ' num2str(Error_B1)]);

%% feature transform through NN
load('temp/C1.mat');
nnT_X = nnTransform(nn_C1, X);
nnT_X_test = nnTransform(nn_C1, X_test);
%% Twin Gaussian Processes with NN transformed features
[InvIK, InvOK] = TGPTrain(nnT_X, Y, Param);
predicted_f0_C = TGPTest(nnT_X_test, nnT_X, Y, Param, InvIK, InvOK);
[Error_C1, TGPErrorvec] = JointError(predicted_f0_C, Y_test);
disp(['TGP: ' num2str(Error_C1)]);

%% feature transform through NN
load('temp/D1.mat');
nnT_X = nnTransform(nn_D1, X);
nnT_X_test = nnTransform(nn_D1, X_test);
%% Twin Gaussian Processes with NN transformed features
[InvIK, InvOK] = TGPTrain(nnT_X, Y, Param);
predicted_f0_D = TGPTest(nnT_X_test, nnT_X, Y, Param, InvIK, InvOK);
[Error_D1, TGPErrorvec] = JointError(predicted_f0_D, Y_test);
disp(['TGP: ' num2str(Error_D1)]);

save('temp/error_TGP.mat','Error_A1','Error_B1','Error_C1','Error_D1','predicted_f0_A','predicted_f0_B','predicted_f0_C','predicted_f0_D');
%%

%-----------------------------GP---------------------------------

load('temp/A1.mat');
nnT_X = nnTransform(nn_A1, X);
nnT_X_test = nnTransform(nn_A1, X_test);
%% Gaussian Process
covfunc = @covSEiso; 
nu = fix(size(nnT_X,1)/2); 
iu = randperm(size(nnT_X,1)); 
iu = iu(1:nu); 
u = nnT_X(iu,:);
covfuncF = {@covFITC, {covfunc},u};
likfunc = @likGauss; 
sn = 0.1; 
hyp2.cov = [0 ; 0];    
hyp2.lik = log(sn);
predicted_f01=zeros(size(Y_test,1),size(Y_test,2));
for i=1:size(Y,2)
    hyp2 = minimize(hyp2, @gp, -100, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i));
    exp(hyp2.lik)
    nlml2 = gp(hyp2, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i))
    [m s2] = gp(hyp2, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i), nnT_X_test);
    predicted_f01(:,i)= m;
end
[Error_A1, TGPErrorvec] = JointError(predicted_f01, Y_test);
disp(['GP_A: ' num2str(Error_A1)]);

%% feature transform through NN
load('temp/B1.mat');
nnT_X = nnTransform(nn_B1, X);
nnT_X_test = nnTransform(nn_B1, X_test);
%% Gaussian Process
covfunc = @covSEiso; 
nu = fix(size(nnT_X,1)/2); 
iu = randperm(size(nnT_X,1)); 
iu = iu(1:nu); 
u = nnT_X(iu,:);
covfuncF = {@covFITC, {covfunc},u};
likfunc = @likGauss; 
sn = 0.1; 
hyp2.cov = [0 ; 0];    
hyp2.lik = log(sn);
predicted_f02=zeros(size(Y_test,1),size(Y_test,2));
for i=1:size(Y,2)
    hyp2 = minimize(hyp2, @gp, -100, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i));
    exp(hyp2.lik)
    nlml2 = gp(hyp2, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i))
    [m s2] = gp(hyp2, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i), nnT_X_test);
    predicted_f02(:,i)= m;
end
[Error_B1, TGPErrorvec] = JointError(predicted_f02, Y_test);
disp(['GP_B: ' num2str(Error_B1)]);

%% feature transform through NN
load('temp/C1.mat');
nnT_X = nnTransform(nn_C1, X);
nnT_X_test = nnTransform(nn_C1, X_test);
%% Gaussian Process
covfunc = @covSEiso; 
nu = fix(size(nnT_X,1)/2); 
iu = randperm(size(nnT_X,1)); 
iu = iu(1:nu); 
u = nnT_X(iu,:);
covfuncF = {@covFITC, {covfunc},u};
likfunc = @likGauss; 
sn = 0.1; 
hyp2.cov = [0 ; 0];    
hyp2.lik = log(sn);
predicted_f03=zeros(size(Y_test,1),size(Y_test,2));
for i=1:size(Y,2)
    hyp2 = minimize(hyp2, @gp, -100, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i));
    exp(hyp2.lik)
    nlml2 = gp(hyp2, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i))
    [m s2] = gp(hyp2, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i), nnT_X_test);
    predicted_f03(:,i)= m;
end
[Error_C1, TGPErrorvec] = JointError(predicted_f03, Y_test);
disp(['GP_C: ' num2str(Error_C1)]);

%% feature transform through NN
load('temp/D1.mat');
nnT_X = nnTransform(nn_D1, X);
nnT_X_test = nnTransform(nn_D1, X_test);
%% Gaussian Process
covfunc = @covSEiso; 
nu = fix(size(nnT_X,1)/2); 
iu = randperm(size(nnT_X,1)); 
iu = iu(1:nu); 
u = nnT_X(iu,:);
covfuncF = {@covFITC, {covfunc},u};
likfunc = @likGauss; 
sn = 0.1; 
hyp2.cov = [0 ; 0];    
hyp2.lik = log(sn);
predicted_f04=zeros(size(Y_test,1),size(Y_test,2));
for i=1:size(Y,2)
    hyp2 = minimize(hyp2, @gp, -100, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i));
    exp(hyp2.lik)
    nlml2 = gp(hyp2, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i))
    [m s2] = gp(hyp2, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i), nnT_X_test);
    predicted_f04(:,i)= m;
end
[Error_D1, TGPErrorvec] = JointError(predicted_f04, Y_test);
disp(['GP_D: ' num2str(Error_D1)]);

save('temp/gp_error.mat','Error_A1','Error_B1','Error_C1','Error_D1','predicted_f01','predicted_f02','predicted_f03','predicted_f04', 'Y_test','X_test');

% %%
% %------------------------------------DNN------------------
% 
% load('temp/A1.mat');
% [er_C1, bad] = nntest(nn_A1, X_test, test_y);
% nn = nnff(nn_A1, X_test, zeros(size(X_test,1), nn.size(end)));
