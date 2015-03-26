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
%% feature transform through NN
load('temp/A1.mat');
nnT_X = nnTransform(nn_A1, X);
nnT_X_test = nnTransform(nn_A1, X_test);
%%
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
predicted_f0=zeros(size(Y_test,1),size(Y_test,2));
for i=1:size(Y,2)
    hyp2 = minimize(hyp2, @gp, -100, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i));
    exp(hyp2.lik)
    nlml2 = gp(hyp2, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i))
    [m s2] = gp(hyp2, @infFITC, [], covfuncF, likfunc, nnT_X, Y(:,i), nnT_X_test);
    predicted_f0(:,i)= m;
end
%%
[Error_A1, TGPErrorvec] = JointError(predicted_f0, Y_test);
disp(['TGP: ' num2str(Error_A1)]);