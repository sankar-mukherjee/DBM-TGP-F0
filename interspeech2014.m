%%
% A 200 200 200
% B 350 350 450
% C 350 300 450
% D 150 150 150
%%
load('data/regression_269.mat');

%% 269 feature
load('weights/dbn_NN_Class269_200_200_200.mat');
nn = dbnunfoldtonn(dbn, 3);
nn.activation_function = 'sigm';
nn.output = 'sigm';
opts.numepochs = 10;
opts.batchsize = 110;
opts.plot               = 1;
nn_A = nntrain(nn, train_nnx, train_nny, opts,val_x,val_y);
[er_A, bad] = nntest(nn_A, test_x, test_y);
save('temp/A.mat', 'nn_A');
%%
load('weights/dbn_NN_Class269_350_350_450.mat');
nn = dbnunfoldtonn(dbn, 3);
nn.activation_function = 'sigm';
nn.output = 'sigm';
opts.numepochs = 10;
opts.batchsize = 110;
opts.plot               = 1;
nn_B = nntrain(nn, train_nnx, train_nny, opts,val_x,val_y);
[er_B, bad] = nntest(nn_B, test_x, test_y);
save('temp/B.mat', 'nn_B');
%%
load('weights/dbn_NN_Class269_350_300_450.mat');
nn = dbnunfoldtonn(dbn, 3);
nn.activation_function = 'sigm';
nn.output = 'sigm';
opts.numepochs = 10;
opts.batchsize = 110;
opts.plot               = 1;
nn_C = nntrain(nn, train_nnx, train_nny, opts,val_x,val_y);
[er_C, bad] = nntest(nn_C, test_x, test_y);
save('temp/C.mat', 'nn_C');

%%
load('weights/dbn_NN_Class269_150_150_150.mat');
nn = dbnunfoldtonn(dbn, 3);
nn.activation_function = 'sigm';
nn.output = 'sigm';
opts.numepochs = 10;
opts.batchsize = 110;
opts.plot               = 1;
nn_D = nntrain(nn, train_nnx, train_nny, opts,val_x,val_y);
[er_D, bad] = nntest(nn_D, test_x, test_y);
save('temp/D.mat', 'nn_D');

%% 351 feature
load('data/regression_351.mat');

load('weights/dbn_NN_Class_200_200_200.mat');
nn = dbnunfoldtonn(dbn, 3);
nn.activation_function = 'sigm';
nn.output = 'sigm';
opts.numepochs = 10;
opts.batchsize = 110;
opts.plot               = 1;
nn_A1 = nntrain(nn, train_nnx, train_nny, opts,val_x,val_y);
[er_A1, bad] = nntest(nn_A1, test_x, test_y);
save('temp/A1.mat', 'nn_A1');
%%
load('weights/dbn_NN_Class_350_350_450.mat');
nn = dbnunfoldtonn(dbn, 3);
nn.activation_function = 'sigm';
nn.output = 'sigm';
opts.numepochs = 10;
opts.batchsize = 110;
opts.plot               = 1;
nn_B1 = nntrain(nn, train_nnx, train_nny, opts,val_x,val_y);
[er_B1, bad] = nntest(nn_B1, test_x, test_y);
save('temp/B1.mat', 'nn_B1');
%%
load('weights/dbn_NN_Class_350_300_450.mat');
nn = dbnunfoldtonn(dbn, 3);
nn.activation_function = 'sigm';
nn.output = 'sigm';
opts.numepochs = 10;
opts.batchsize = 110;
opts.plot               = 1;
nn_C1 = nntrain(nn, train_nnx, train_nny, opts,val_x,val_y);
[er_C1, bad] = nntest(nn_C1, test_x, test_y);
save('temp/C1.mat', 'nn_C1');

%%
rng(0);
dbn.sizes = [150 150 150];
opts.numepochs =   20;
opts.batchsize = 10;
opts.momentum  =   0.95;
opts.alpha     =   0.0002;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

nn = dbnunfoldtonn(dbn, 3);
nn.activation_function = 'sigm';
nn.output = 'sigm';
opts.numepochs = 10;
opts.batchsize = 110;
opts.plot               = 1;
nn_D1 = nntrain(nn, train_nnx, train_nny, opts,val_x,val_y);
[er_D1, bad] = nntest(nn_D1, test_x, test_y);
save('temp/D1.mat', 'nn_D1');

%%
save('temp/error.mat','er_A','er_B','er_C','er_D','er_A1','er_B1','er_C1','er_D1');
