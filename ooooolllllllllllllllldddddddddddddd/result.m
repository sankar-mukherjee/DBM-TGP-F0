
%%
%load('saved_weights/dbn_NN_Class_R_3L_450_99m_0002alpha.mat');
%load('temp_dbn_dnn.mat');
sentence_id = 10337;
[sen_i, ~] = find(sen_index(:,2)==sentence_id);
sen_dur = sen_index(sen_i:sen_i+1,1);
[w_i1,~] = find(word_index==sen_dur(1));
[w_i2,~] = find(word_index==sen_dur(2));
word_i = word_index(w_i1:w_i2-1);
sen_dur(2) = sen_dur(2)-1;

input_text = input((sen_dur(1)-1)*5+1:(sen_dur(2)-1)*5+5,:); 
o = f0_5state((sen_dur(1)-1)*5+1:(sen_dur(2)-1)*5+5,:);

%% NN predicted f0 delta and delta-delta
f0 = []; d_f0=[];dd_f0=[];
labels_f = nnpredict(nn_f, input_text);
labels_d = nnpredict(nn_d, input_text);     z_d=unique(ceil(delta));            z_d=reshape(z_d,[10 80])';      % 80 class delta
labels_dd = nnpredict(nn_dd, input_text);   z_dd=unique(ceil(delta_delta));     z_dd=reshape(z_dd,[8 139])';    % 139 class delta-delta
for i=1:size(input_text,1)
    % ----------f0------------------------
    a=labels_f(i,1);
    if(a==1)
        a=0;
    else
        a=a+73;
    end
    f0 = [f0;a];
    % ----------d------------------------
    a=labels_d(i,1);
    a = median(z_d(a,:));
    d_f0 = [d_f0;a];
    % ----------dd------------------------
    a=labels(i,1);
    a = median(z_dd(a,:));
    dd_f0 = [dd_f0;a];
end
%% for delta and delta delta feature combination
target_f0 = add_delta_deltadelta(f0,d_f0,dd_f0);
target_f0(target_f0>0)=0;
%%
figure;plot(o);hold on;plot(-target_f0,'r');hold off;
rmse = rmse(o,-target_f0)
xcor = xcorr(o,-target_f0,0,'coeff')
