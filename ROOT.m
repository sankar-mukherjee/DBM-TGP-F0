% DBM-TGP
clear;clc;
%% Controll experiment
F_loaddata = 0;
F_dataprep = 0;
F_DBN = 0;
F_DBM = 0;
F_TGP = 0;
F_result_DBNTGP = 0;
F_rmse_xcorr = 0;
F_SYN = 0;

addpath('TGP');
addpath('DBM-kcho');
addpath('utility');
%% prepare class data
if F_dataprep
    data_preprocess_class
end
%% DBN / DBM
if F_DBN
    DBN
end
if F_DBM
    DBM
end
%% Twin Gaussin Process
if F_TGP    
    TGP
end
%% Result_analysis
% DBN TGP
if F_result_DBNTGP    
    result_DBNTGP    
end
% DBN
if F_result_DBN    
    result_DBN  
end
% DBM TGP

%% RMSE XCORR
if F_rmse_xcorr 
    RMSE_XCORR  
end
%% Synthesis
if F_SYN   
    SYNTHESIS;
end
