% Deep Belief Netwrok
load('data/regression_269.mat');
load('weights/dbn_NN_Class269_200_200_200.mat');

%%
% rng(0);
% dbn.sizes = [200 200 200];
% opts.numepochs =   20;
% opts.batchsize = 10;
% opts.momentum  =   0.95;
% opts.alpha     =   0.0002;
% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);

%% nn for f0, delta, delta-delta
nn = dbnunfoldtonn(dbn, 3);
nn.activation_function = 'sigm';
nn.output = 'softmax';
opts.numepochs = 3;
opts.batchsize = 110;
%train nn
% nn.learningRate  = 0.0002;
% nn.dropoutFraction = 0.5;
% nn.weightPenaltyL2 = 1e-4;
opts.plot               = 1; 

nn_1 = nntrain(nn, train_nnx, train_nny, opts,val_x,val_y);
[er_f1, bad] = nntest(nn_1, test_x, test_y);

nn.output = 'sigm';
nn_2 = nntrain(nn, train_nnx, train_nny, opts,val_x,val_y);
[er_f2, bad] = nntest(nn_2, test_x, test_y);

nn.output = 'linear';
nn_3 = nntrain(nn, train_nnx, train_nny, opts,val_x,val_y);
[er_f3, bad] = nntest(nn_3, test_x, test_y);