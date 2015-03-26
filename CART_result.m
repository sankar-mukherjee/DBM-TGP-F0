%% result of pedicted to the original f0 by CART
addpath('utility');

filelist_tag = dir('../BIG_HTS_sentences_for_test/0/*.wav');
for ss=1:length(filelist_tag)
    path = ['../BIG_HTS_sentences_for_test/0/' num2str(filelist_tag(ss,1).name)];    
    [x,fs]=wavread(path);
    [cart_f0,~,~]=exstraightsource(x,fs);
    pitch_path = ['../Deep_F0/pitch/' strrep(num2str(filelist_tag(ss,1).name),'.wav','.txt')];
    original_f0 = load(pitch_path);
    %% return result
    result_return{ss,1} = original_f0;
    result_return{ss,2} = cart_f0;
end