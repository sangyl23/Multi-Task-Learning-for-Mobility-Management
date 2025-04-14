%{
 @ File: Generator.m
 @ Time: 2024/05/18 
 @ Author : Yiliang Sang
 @ Description: Channel data, BS location and UE location generator
%}
clc; clear; close all;

addpath('DeepMIMO_functions')
params = read_params('parameters.m');

% parameters
row_num = params.active_user_last - params.active_user_first + 1;
UE_row_num = 181;
BS_num = length(params.active_BS);
antenna_num = params.num_ant_BS(2);
fprintf('Row_first:%d, Row_last:%d\n', params.active_user_first, params.active_user_last);

% candidate mmWave channel
MM_channel = zeros(BS_num, UE_row_num, antenna_num);
UE_loc = zeros(UE_row_num, 2); % only care x-y axis, and omit z axis
BS_loc = zeros(BS_num, 2);

active_user_first0 = params.active_user_first;
% for each row
for i = 1 : row_num
    % select active users (this row)
    params.active_user_first = i + active_user_first0 - 1;
    params.active_user_last = i + active_user_first0 - 1;
    print_info = ['mmWave dataset ' num2str(i) 'th row generation started']
    % generate corresponding channels and parameters
    [dataset_MM, params_MM] = DeepMIMO_generator(params);
    % save channels into matrices
    for BS_idx = 1 : BS_num
        for k = 1 : UE_row_num  
            MM_channel(BS_idx, k, :) = squeeze(sum(dataset_MM{BS_idx}.user{k}.channel, 3));
            UE_loc(k, :) = dataset_MM{BS_idx}.user{k}.loc(1 : 2);
        end
        BS_loc(BS_idx, :) = dataset_MM{BS_idx}.loc(1 : 2);
    end

    % save channels into files
    fprintf('\n Saving the DeepMIMO Dataset ...')
    % sfile_DeepMIMO = ['./DeepMIMO_BS1-4-8-9_Row250-2500_dataset/dataset_row' num2str(i + active_user_first0 - 1) '.mat'];
    sfile_DeepMIMO = ['./DeepMIMO_BS1-10-14-17_Row1400-1650_dataset/dataset_row' num2str(i + active_user_first0 - 1) '.mat'];
    save(sfile_DeepMIMO, 'MM_channel', 'UE_loc', 'BS_loc');
    % MM_channel: size (BS_num, UE_row_num, antenna_num)
    % UE_loc: size (UE_row_num, 2)
    % BS_loc: size (BS_num, 2), actually remains unchanged with i
end
fprintf('\n DeepMIMO Dataset Generation completed \n')

