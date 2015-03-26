%% Result analysis through RMSE and XCORR
addpath('utility');

rmse = 0; xcor = 0;
for ii = 1:size(result_return,1)
    rmse = rmse + rmse(result_return{ii,1},result_return{ii,2});
    xcor = xcor + xcorr(result_return{ii,1},result_return{ii,2},0,'coeff');
end
rmse = rmse / size(result_return,1)
xcor = xcor / size(result_return,1)