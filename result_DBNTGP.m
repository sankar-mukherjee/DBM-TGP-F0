%% result of pedicted to the original f0 by DBN-TGP
addpath('utility');

load('data/sen_index.mat');load('data/word_index.mat');load('data/syn_duration.mat');
load('data/input_269.mat');load('data/f0.mat');load('data/delta_f0.mat');
load('data/test_sentence.mat');
load('data/scaling_factor.mat');
%%
X = train_nnx(perm_idx, :);
Y = train_nny(perm_idx, :);
X_test = test_x(1:150, :);
Y_test = test_y(1:150, :);
%% DBN --> NN and feature transform through NN    
nn = dbnunfoldtonn(dbn, 327);       % 327 is meannig less only dbn --> nn structure
nnT_X = nnTransform(nn, X);
%% TGP parameters
Param.kparam1 = 2*1e-4;
Param.kparam2 = 2*1e-6;
Param.lambda = 1e-3;

for ss=1:size(test_sentence,1)
    sentence_id =  str2double(cell2mat(test_sentence(ss,1)));
    [sen_i, ~] = find(sen_index(:,2)==sentence_id);
    sen_dur = sen_index(sen_i:sen_i+1,1);
    [w_i1,~] = find(word_index==sen_dur(1));
    [w_i2,~] = find(word_index==sen_dur(2));
    word_i = word_index(w_i1:w_i2-1);
    sen_dur(2) = sen_dur(2)-1;
    %% input / output data preparation
    input_text = input((sen_dur(1)-1)*5+1:(sen_dur(2)-1)*5+5,:);
    original = f0_5state((sen_dur(1)-1)*5+1:(sen_dur(2)-1)*5+5,:);
    %% feature transform through NN    
    X_test = nnTransform(nn, input_text);
    %% prediction through TGP    
    predicted = TGPTest(X_test, nnT_X, Y, Param, InvIK, InvOK);    
    %% scale up the predicted values 
    predicted(:,1) = (predicted(:,1) -0.1)/(0.99 - 0.1);
    predicted(:,2) = (predicted(:,2) -0.1)/(0.99 - 0.1);
    predicted(:,3) = (predicted(:,3) -0.1)/(0.99 - 0.1);    
    predicted(:,1) = predicted(:,1)*scaling_factors(1,1)+scaling_factors(1,2);
    predicted(:,2) = predicted(:,2)*scaling_factors(2,1)+scaling_factors(2,2);
    predicted(:,3) = predicted(:,3)*scaling_factors(3,1)+scaling_factors(3,2);
    %% for delta and delta delta feature combination
    target_f0 = add_delta_deltadelta(predicted(:,1),predicted(:,2),predicted(:,3));
    target_f0(target_f0>0)=0;
    %% return result
    result_return{ss,1} = original;
    result_return{ss,2} = -target_f0;
end

