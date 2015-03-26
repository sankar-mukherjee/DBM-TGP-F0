

load('temp/A1.mat');
nnT_X = nnTransform(nn_A1, X);
nnT_X_test = nnTransform(nn_A1, input_text);
%% Twin Gaussian Processes with NN transformed features
[InvIK, InvOK] = TGPTrain(nnT_X, Y, Param);
predicted = TGPTest(nnT_X_test, nnT_X, Y, Param, InvIK, InvOK);

target_f0 = -add_delta_deltadelta(predicted(:,1),predicted(:,2),predicted(:,3));
ori = add_delta_deltadelta(original(:,1),original(:,2),original(:,3));
ori(ori<0)=0;
mse(target_f0,ori)
xcorr(target_f0,ori,0,'coeff')
figure;plot(target_f0);figure(gcf);hold on;plot(ori,'r')

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
target_f01 = -add_delta_deltadelta(predicted_f0(:,1),predicted_f0(:,2),predicted_f0(:,3));
ori = add_delta_deltadelta(original(:,1),original(:,2),original(:,3));
ori(ori<0)=0;
mse(target_f01,ori)
xcorr(target_f01,ori,0,'coeff')
figure;plot(target_f01);figure(gcf);hold on;plot(ori,'r')