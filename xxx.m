
[Error_A1, ~] = JointError(predicted, Y_test)
xc=0;
for ii=1:3
xc = xc+xcorr(predicted(:,ii),Y_test(:,ii),0,'coeff');
end
xc/3
train_idx = randperm (10000);
test_idx = randperm (3000);
myVars = {'test_x','test_y'};

%% 351 feature
load('data/regression_351.mat',myVars{:});
X_test = test_x(test_idx, :);
Y_test = test_y(test_idx, :);

load('temp/D1.mat');
nn = nnff(nn_D1, input_text, zeros(size(input_text,1), nn_D1.size(end)));
predicted = nn.e;
target_f0 = -add_delta_deltadelta(predicted(:,1),predicted(:,2),predicted(:,3));
ori = add_delta_deltadelta(original(:,1),original(:,2),original(:,3));
ori(ori<0)=0;
mse(target_f0,ori)
xcorr(target_f0,ori,0,'coeff')
plot(target_f0);figure(gcf);hold on;plot(ori,'r')
%%
load('temp/A1.mat');
nn = nnff(nn_A1, X_test, zeros(size(X_test,1), nn_A1.size(end)));
predicted = nn.e;
target_f0 = add_delta_deltadelta(predicted(:,1),predicted(:,2),predicted(:,3));
ori = add_delta_deltadelta(Y_test(:,1),Y_test(:,2),Y_test(:,3));
mse(target_f0,ori)
xcorr(target_f0,ori,0,'coeff')

%%
load('temp/B1.mat');
nn = nnff(nn_B1, X_test, zeros(size(X_test,1), nn_B1.size(end)));
predicted = nn.e;
target_f0 = add_delta_deltadelta(predicted(:,1),predicted(:,2),predicted(:,3));
ori = add_delta_deltadelta(Y_test(:,1),Y_test(:,2),Y_test(:,3));
mse(target_f0,ori)
xcorr(target_f0,ori,0,'coeff')

%%
load('temp/C1.mat');
nn = nnff(nn_C1, X_test, zeros(size(X_test,1), nn_C1.size(end)));
predicted = nn.e;
target_f0 = add_delta_deltadelta(predicted(:,1),predicted(:,2),predicted(:,3));
ori = add_delta_deltadelta(Y_test(:,1),Y_test(:,2),Y_test(:,3));
mse(target_f0,ori)
xcorr(target_f0,ori,0,'coeff')

%%
load('temp/D1.mat');
nn = nnff(nn_D1, X_test, zeros(size(X_test,1), nn_D1.size(end)));
predicted = nn.e;
target_f0 = add_delta_deltadelta(predicted(:,1),predicted(:,2),predicted(:,3));
ori = add_delta_deltadelta(Y_test(:,1),Y_test(:,2),Y_test(:,3));
mse(target_f0,ori)
xcorr(target_f0,ori,0,'coeff')