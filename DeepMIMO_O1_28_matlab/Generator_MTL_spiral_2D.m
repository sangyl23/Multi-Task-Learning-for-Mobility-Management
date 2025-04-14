clear all;
close all;
clc;

addpath('DeepMIMO_functions')
params = read_params('parameters.m');

% MTL
% DeepMIMO, O1
% 28GHz
% row: 250-2500

% MM antenna num
M_TX = params.num_ant_BS(2);
% MM narrow beam num
MM_narrow_beam_num = M_TX;
UE_row_num = 181;
BS_num = length(params.active_BS);
% angular range
sector_start = - pi;
sector_end = pi;
% narrow beam generation
candidate_narrow_beam_angle = sector_start + (sector_end - sector_start) / MM_narrow_beam_num * [0.5 : 1 : MM_narrow_beam_num - 0.5];
candidate_narrow_beam = exp(-1i * [0 : M_TX - 1]' * candidate_narrow_beam_angle) / sqrt(MM_narrow_beam_num); % (M_TX, beam_num)

% UE distribution
% row_index = [1 : params.active_user_last - params.active_user_first + 1];
row_begin_idx = 1400;
row_end_idx = 1650;
row_index = [row_begin_idx : row_end_idx];

MM_ch = zeros(BS_num, UE_row_num, length(row_index), MM_narrow_beam_num);
UE_loc_all = zeros(UE_row_num, length(row_index), 2);
BS_loc_all = zeros(BS_num, 2);

active_user_first0 = params.active_user_first;

for i = row_index
    read_idx = i;
    write_idx = i - row_begin_idx + 1;
    MM_file = ['./DeepMIMO_BS1-10-14-17_Row1400-1650_dataset/dataset_row' num2str(read_idx) '.mat'];
    load(MM_file);

    % MM_channel: size (BS_num, UE_row_num, antenna_num)
    % UE_loc: size (UE_row_num, 2)
    % BS_loc: size (BS_num, 2) 
    MM_channel = permute(MM_channel, [3, 1, 2]); % (antenna_num, BS_num, UE_row_num)
    tmp(1, :, :, :) = MM_channel; % (1, antenna_num, BS_num, UE_row_num)  
    tmp = pagemtimes(tmp, candidate_narrow_beam); % (1, beam_num, BS_num, UE_row_num)
    tmp = squeeze(tmp); % (beam_num, BS_num, UE_row_num)
    MM_ch(:, :, write_idx, :) = permute(tmp, [2, 3, 1]); % (BS_num, UE_row_num, beam_num)    
    UE_loc_all(:, write_idx, :) = UE_loc;
    BS_loc_all = BS_loc; 
    clear tmp;
end 

% file number
file_num = 550;
% sample number in each file
batch_num = 256;
% UE speed
% speeds = [10]; % m/s

% 2D spiral parameters
omega_min = -0.5; % angular velocity (rad/s)
omega_max = 0.5;
c_min = -10 / 2 / pi; % radial growth rate (m/s)
c_max = 10 / 2 / pi;

% length of historical sequence 
his_len = 9;
% length of predicted sequence
pre_len = 1;

% MM beam training received signal
MM_data = zeros(batch_num, 2, his_len + pre_len, BS_num, MM_narrow_beam_num);
% optimal BS idx
BS_label = zeros(batch_num, his_len + pre_len);
% MM optimal beam index
beam_label = zeros(batch_num, his_len + pre_len);
% MM beam amplitude
beam_power = zeros(batch_num, his_len + pre_len, BS_num, MM_narrow_beam_num);
% UE location
UE_loc_data = zeros(batch_num, his_len + pre_len, 2);
% BS_location
BS_loc_data = zeros(batch_num, his_len + pre_len, BS_num, 2);

for i = 1 : file_num
    for j = 1 : batch_num
        % find UE trajectory within the pre-defined range
        flag = 0;
        while flag == 0
            initial_x = round(rand * 181); % U(0, 181)
            initial_y = round(1425 + rand * 200); % U(1425, 1625) for Row1400-1650
            
            t_list = [0 : 0.16 : 1.6 - 0.16].';
            x_mid = 181 / 2;
            y_mid = 1425 + 200 / 2;
            current_omega = omega_min + (omega_max - omega_min) * rand;
            current_c = c_min + (c_max - c_min) * rand;
            location = zeros(size(t_list, 1), 2);

            [location(:, 1), location(:, 2)] = spiral_motion(initial_x, initial_y, x_mid, y_mid, current_omega, current_c, t_list);

            if  min(location(:, 1)) >= 1 && max(location(:, 1)) <= UE_row_num && ...
                min(location(:, 2)) >= row_begin_idx && max(location(:, 2)) <= row_end_idx
                flag = 1;
            end
        end

        % save corresponding data
        % MM_data: sequence of mmWave beam training received signal, (b, 2, his_len + pre_len, BS_num, beam_num)
        % BS_label: sequence of optimal BS idx, (b, his_len + pre_len)
        % beam_label: sequence of mmWave optimal beam, (b, his_len + pre_len)
        % beam_power: sequence of mmWave beam training received signal,
        % (b, his_len + pre_len, BS_num, beam_num)
        location(:, 2) = location(:, 2) - row_begin_idx + 1; % (his_len + pre_len, 2)
        % location = repmat([180; 1100], 1, his_len + pre_len)
        % location = location.'
        for k = 1 : his_len + pre_len           
            tmp = squeeze(MM_ch(:, location(k, 1), location(k, 2), :)); % (BS_num, beam_num)
            [~, beam_label(j, k)] = max(reshape(abs(tmp.'), MM_narrow_beam_num * BS_num, 1));
            BS_label(j, k) = floor((beam_label(j, k) - 1) / MM_narrow_beam_num) + 1;
            MM_data(j, 1, k, :, :) = real(tmp);
            MM_data(j, 2, k, :, :) = imag(tmp);
            beam_power(j, k, :, :) = abs(tmp);

            UE_loc_data(j, k, :) = UE_loc_all(location(k, 1), location(k, 2), :);
            BS_loc_data(j, k, :, :) = BS_loc_all;
            clear tmp;
        end

        % add noise for receive signal
        SNR = 5; % dB
        for BS_idx = 1 : BS_num
            tmp = squeeze(MM_data(j, 1, :, BS_idx, :)) + 1j * squeeze(MM_data(j, 2, :, BS_idx, :));
            tmp = awgn(tmp, SNR, 'measured');
            MM_data(j, 1, :, BS_idx, :) = real(tmp);
            MM_data(j, 2, :, BS_idx, :) = imag(tmp);
        end

    end

    % save channels into files
    fprintf(['\n Saving the MTL Dataset', num2str(i)])
    save_filename = ['./Spiral2D_BS1-10-14-17_Row1400-1650_MTLdataset/snr' num2str(SNR) 'dB/train_500mats/dataset_' num2str(i) '.mat'];
    save(save_filename, 'MM_data', 'BS_label', 'beam_label', 'beam_power', 'UE_loc_data', 'BS_loc_data');

    % MM_data: receive signal with size (batch, 2, his_len + pre_len, BS_num, beam_num)
    % BS_label: optimal BS idx with size (batch, his_len + pre_len)
    % beam_label: optimal beam with size (batch, his_len + pre_len)
    % beam_power: power of receive signal with size (batch, his_len + pre_len, BS_num, beam_num)
    % UE_loc_data: UE location with size (batch, his_len + pre_len, 2)
    % BS_loc_data: main BS location with size (batch, his_len + pre_len, BS_num, 2)
end
