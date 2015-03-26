%% Speech Synthesis through MLSA with HTS duration

sentence_id = 10337;

load('data/syn_duration.mat');
%% make f0 according to HTS duration 
ind=find(ismember(dur_name,num2str(sentence_id)));
phoneme = ph_dur(ind,:)';
duration = cell2mat(R_dur(ind,:))';
phoneme = phoneme(~cellfun('isempty',phoneme)); 
state_duration=reshape(duration,5,size(duration,1)/5)';
f0=[];
state_f0=reshape(target_f0,5,size(target_f0,1)/5)';
for d=1:size(phoneme,1)
    if(strcmp(phoneme(d),'sil'))
        f0 = [f0; zeros(sum(state_duration(d,:)),1)];
    elseif(strcmp(phoneme(d),'sp'))
        f0 = [f0; zeros(sum(state_duration(d,:)),1)];
    else
        f = state_f0(1,:);
        dur = state_duration(d,:);
        for s=1:5
            f0 = [f0; repmat(f(s),[dur(s) 1])];
        end
        state_f0 = state_f0(2:end,:);
    end
end
%% put the generated f0 file in MLSA/gen/ and with .mgc file synthesize speech
path = ['MLSA/gen/' num2str(sentence_id)];
dlmwrite(path,f0);
perl ('mlsa.pl');
