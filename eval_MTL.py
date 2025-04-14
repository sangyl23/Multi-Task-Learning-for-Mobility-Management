import torch.optim as optim
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from dataloader_MTL import Dataloader
from model_MTL import Bs, bt, Vanilla, Bs2bt2Up, Up2bt2Bs, Dual_Cascaded
import sys
import time
import argparse
import logging
import matplotlib
import matplotlib.pyplot as plt
import os

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("logfile.txt", mode = 'w'),
    logging.StreamHandler()
])
logger = logging.getLogger()

parser = argparse.ArgumentParser()

# scenario parameters
parser.add_argument('--experiment_type', type = str, default = 'O1_trainingsamples',
                    choices = ['O1_trainingsamples', 'O1_velocity', 'O1_snr', 'O1_motion_form', 'Outdoor_Blockage_trainingsamples'])
parser.add_argument('--visualize', type = eval, default = True)
parser.add_argument('--save_path', type = str, default = 'results')


args = parser.parse_args()

def eval_Bs(model, loader, b, his_len, pre_len, BS_num, beam_num, device):
    # reset dataloader
    loader.reset()
    # judge whether dataset is finished
    done = False
    # top1_acc
    BS_top1_acc = 0.
    # count batch number
    batch_num = 0
    
    with torch.no_grad():
        # evaluate validation set
        while True:
            batch_num += 1
            # read files
            # channels: sequence of mmWave beam training received signal vectors with size (b, 2, his_len + pre_len, BS_num, beam_num)
            # BS_labels: sequence of optimal BS idx with size (b, his_len + pre_len)
            channels, BS_label, _, _, _, _, done = loader.next_batch()
            
            if done == True:
                break

            # select data for BS selection
            channel_his = channels[:, :, 0 : his_len : 1, :, :] # (b, 2, his_len, BS_num, beam_num)
            BS_label_pre = BS_label[:, his_len] # (b)
            BS_label_pre = BS_label_pre.to(torch.int64)

            # predicted results
            out_tensor = model(channel_his) # (b, BS_num)

            BS_top1_acc += (torch.sum(torch.argmax(out_tensor, dim = 1) == BS_label_pre) / b).item()
            
    # average accuracy
    BS_acur = BS_top1_acc / batch_num

    return BS_acur

def eval_bt(model, loader, b, his_len, pre_len, BS_num, beam_num, device):
    # reset dataloader
    loader.reset()
    # judge whether dataset is finished
    done = False
    # top1_acc
    # BS_top1_acc = 0.
    beam_top1_acc = 0.
    beam_norm_gain = 0.
    # count batch number
    batch_num = 0
    
    with torch.no_grad():
        # evaluate validation set
        while True:
            batch_num += 1
            # read files
            # channels: sequence of mmWave beam training received signal vectors with size (b, 2, his_len + pre_len, BS_num, beam_num)
            # beam_label: sequence of optimal beam idx with size (b, his_len + pre_len)
            # beam_power: sequence of beam power (b, his_len + pre_len, BS_num, beam_num)
            channels, _, beam_label, beam_power, _, _, done = loader.next_batch()
            
            if done == True:
                break

            # select data for BS selection
            channel_his = channels[:, :, 0 : his_len : 1, :, :] # (b, 2, his_len, BS_num, beam_num)
            # BS_label_pre = BS_label[:, his_len] # (b)
            # BS_label_pre = BS_label_pre.to(torch.int64)
            beam_label_pre = beam_label[:, his_len] # (b)
            beam_label_pre = beam_label_pre.to(torch.int64)
            beam_power = beam_power.reshape(b, his_len + pre_len, -1) # (b, his_len + pre_len, BS_num * beam_num)
            beam_power_pre = beam_power[:, his_len, :] # (b, BS_num * beam_num)

            # predicted results
            out_tensor = model(channel_his) # (b, beam_num * BS_num)
            
            # BS_top1_acc += (torch.sum(torch.argmax(out_tensor, dim = 1) == BS_label_pre) / b).item()
            beam_top1_acc += (torch.sum(torch.argmax(out_tensor, dim = 1) == beam_label_pre) / b).item()
            
            gt_beampower_idx = beam_label_pre.unsqueeze(dim = 1) # (b, 1)
            pre_beampower_idx = torch.argmax(out_tensor, dim = 1)
            pre_beampower_idx = pre_beampower_idx.unsqueeze(dim = 1) # (b, 1)
            beam_norm_gain += torch.mean(torch.squeeze(torch.gather(beam_power_pre, 1, pre_beampower_idx), dim = 1) / torch.squeeze(torch.gather(beam_power_pre, 1, gt_beampower_idx), dim = 1))
                    
    # average accuracy
    # BS_acur = BS_top1_acc / batch_num
    beam_acur = beam_top1_acc / batch_num
    beam_norm_gain = beam_norm_gain / batch_num

    return beam_acur, beam_norm_gain

def eval_Up(model, loader, b, his_len, pre_len, BS_num, beam_num, device):
    # reset dataloader
    loader.reset()
    # judge whether dataset is finished
    done = False
    # top1_acc
    # BS_top1_acc = 0.
    # beam_top1_acc = 0.
    # beam_norm_gain = 0.
    # l2 norm distance
    dis_l2 = 0.
    # count batch number
    batch_num = 0
    
    with torch.no_grad():
        # evaluate validation set
        while True:            
            batch_num += 1
            # read files
            # channels: sequence of mmWave beam training received signal vectors with size (b, 2, his_len + pre_len, BS_num, beam_num)
            # UE_loc: sequence of UE position with size (b, his_len + pre_len, 2)
            # BS_loc: sequence of BS position with size (b, his_len + pre_len, BS_num, 2)
            channels, _, _, _, UE_loc, BS_loc, done = loader.next_batch()
            
            if done == True:
                break

            # select data for BS selection
            channel_his = channels[:, :, 0 : his_len : 1, :, :] # (b, 2, his_len, BS_num, beam_num)
            # BS_label_pre = BS_label[:, his_len] # (b)
            # BS_label_pre = BS_label_pre.to(torch.int64)
            # beam_label_pre = beam_label[:, his_len] # (b)
            # beam_label_pre = beam_label_pre.to(torch.int64)
            # beam_power = beam_power.reshape(b, his_len + pre_len, -1) # (b, his_len + pre_len, BS_num * beam_num)
            # beam_power_pre = beam_power[:, his_len, :] # (b, BS_num * beam_num)
            # predicted UE position label
            relative_loc = UE_loc - torch.mean(BS_loc, dim = 2)
            UE_loc_pre = relative_loc[:, his_len - 1, :] # (b, 2)
            
            # predicted results
            out_tensor = model(channel_his) # (b, 2)

            # BS_top1_acc += (torch.sum(torch.argmax(out_tensor, dim = 1) == BS_label_pre) / b).item()
            # beam_top1_acc += (torch.sum(torch.argmax(out_tensor, dim = 1) == beam_label_pre) / b).item()      
            
            # gt_beampower_idx = beam_label_pre.unsqueeze(dim = 1) # (b, 1)
            # pre_beampower_idx = torch.argmax(out_tensor, dim = 1)
            # pre_beampower_idx = pre_beampower_idx.unsqueeze(dim = 1) # (b, 1)
            # beam_norm_gain += torch.mean(torch.squeeze(torch.gather(beam_power_pre, 1, pre_beampower_idx), dim = 1) / torch.squeeze(torch.gather(beam_power_pre, 1, gt_beampower_idx), dim = 1))
            
            dis_l2 += torch.mean(torch.sqrt((out_tensor[:, 0] - UE_loc_pre[:, 0]) ** 2 + (out_tensor[:, 1] - UE_loc_pre[:, 1]) ** 2))
            
    # average accuracy
    # BS_acur = BS_top1_acc / batch_num
    # beam_acur = beam_top1_acc / batch_num
    # beam_norm_gain = beam_norm_gain / 
    dis = dis_l2 / batch_num


    return dis

# model_evaluation
def eval_Bs_bt_Up(model, loader, b, his_len, pre_len, BS_num, beam_num, device):
    # reset dataloader
    loader.reset()
    # judge whether dataset is finished
    done = False
    # top1_acc
    BS_top1_acc = 0.
    beam_top1_acc = 0.
    beam_norm_gain = 0.
    # l2 norm distance
    dis_l2 = 0.
    # count batch number
    batch_num = 0
    
    with torch.no_grad():
        # evaluate validation set
        while True:                        
            batch_num += 1
            # read files
            # channels: sequence of mmWave beam training received signal vectors with size (b, 2, his_len + pre_len, BS_num, beam_num)
            # BS_label: sequence of optimal BS idx with size (b, his_len + pre_len)
            # beam_label: sequence of optimal beam idx with size (b, his_len + pre_len)
            # beam_power: sequence of beam power (b, his_len + pre_len, BS_num, beam_num)
            # UE_loc: sequence of UE position with size (b, his_len + pre_len, 2)
            # BS_loc: sequence of BS position with size (b, his_len + pre_len, BS_num, 2)
            channels, BS_label, beam_label, beam_power, UE_loc, BS_loc, done = loader.next_batch()
            
            if done == True:
                break

            # select data for BS selection
            channel_his = channels[:, :, 0 : his_len : 1, :, :] # (b, 2, his_len, BS_num, beam_num)
            BS_label_pre = BS_label[:, his_len] # (b)
            BS_label_pre = BS_label_pre.to(torch.int64)
            beam_label_pre = beam_label[:, his_len] # (b)
            beam_label_pre = beam_label_pre.to(torch.int64)
            beam_power = beam_power.reshape(b, his_len + pre_len, -1) # (b, his_len + pre_len, BS_num * beam_num)
            beam_power_pre = beam_power[:, his_len, :] # (b, BS_num * beam_num)
            # predicted UE position label
            relative_loc = UE_loc - torch.mean(BS_loc, dim = 2)
            UE_loc_pre = relative_loc[:, his_len - 1, :] # (b, 2)
            
            # predicted results
            out_BS_label, out_beam_label, out_loc = model(channel_his) 
            
            BS_top1_acc += (torch.sum(torch.argmax(out_BS_label, dim = 1) == BS_label_pre) / b).item()
            beam_top1_acc += (torch.sum(torch.argmax(out_beam_label, dim = 1) == beam_label_pre) / b).item()      
            
            gt_beampower_idx = beam_label_pre.unsqueeze(dim = 1) # (b, 1)
            pre_beampower_idx = torch.argmax(out_beam_label, dim = 1)
            pre_beampower_idx = pre_beampower_idx.unsqueeze(dim = 1) # (b, 1)
            beam_norm_gain += torch.mean(torch.squeeze(torch.gather(beam_power_pre, 1, pre_beampower_idx), dim = 1) / torch.squeeze(torch.gather(beam_power_pre, 1, gt_beampower_idx), dim = 1))
            
            dis_l2 += torch.mean(torch.sqrt((out_loc[:, 0] - UE_loc_pre[:, 0]) ** 2 + (out_loc[:, 1] - UE_loc_pre[:, 1]) ** 2))
            
    BS_acur = BS_top1_acc / batch_num
    beam_acur = beam_top1_acc / batch_num
    beam_norm_gain = beam_norm_gain / batch_num
    dis = dis_l2 / batch_num
    
    return BS_acur, beam_acur, beam_norm_gain, dis


# main function for model training and evaluation
# output: accuracy, losses and normalized beamforming gain
def main(training_time = 3, epoch_num = 100, batch_size = 64):
    version_name = 'Multi-task learning for mobility management'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print basic information
    print(version_name)
    print('device:%s' % device)
    print('batch_size:%d' % batch_size)
    
    for arg in vars(args):
        logger.info(f'{arg} = {getattr(args, arg)}')
    
    # system parameters
    b = batch_size
    his_len = 9 # length of historical sequence
    pre_len = 1 # length of predicted sequence
    beam_num = 32 # beam number
        
    
    if args.experiment_type == 'O1_trainingsamples':
        BS_num = 4 # BS number
        path_eval = 'eval_dataset/O1_v10ms_snr5dB_rectilinearmotion'
        # path_eval = 'eval_dataset/tmp'
            
        eval_loader = Dataloader(path = path_eval, batch_size = b, his_len = his_len, pre_len = pre_len, BS_num = BS_num, beam_num = beam_num, device = device)
        
        
        mat_list = ['100mats', '200mats', '300mats', '400mats', '500mats'] # each mat corresponds to 256 data samples
        # mat_list = ['400mats'] # each mat corresponds to 256 data samples
            
        # save results
        BS_acur_eval = np.zeros((5, len(mat_list)))
        beam_acur_eval = np.zeros((5, len(mat_list)))
        beam_norm_gain_eval = np.zeros((5, len(mat_list)))
        dis_eval = np.zeros((5, len(mat_list)))
        
    
        # first loop for training runnings
        for mat_count in range(len(mat_list)):
            print('load: ' + mat_list[mat_count])
             
            print('load dual-cascaded multi-task model!')         
            model_name = 'trained_model/Dual_Cascaded_O1_' + mat_list[mat_count] + '_velocity10ms_snr5dB_rectilinear_motion.pkl'           
            dual_cascaded = torch.load(model_name, map_location = device)           
            dual_cascaded.to(device)
            dual_cascaded.device = device
            dual_cascaded.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(dual_cascaded, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[0, mat_count] = BS_acur
            beam_acur_eval[0, mat_count] = beam_acur
            beam_norm_gain_eval[0, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[0, mat_count] = dis.detach().cpu().numpy()
            
            print('load single-cascaded multi-task model!')         
            model_name = 'trained_model/Bs2bt2Up_O1_' + mat_list[mat_count] + '_velocity10ms_snr5dB_rectilinear_motion.pkl'           
            Bs2bt2Up = torch.load(model_name, map_location = device)           
            Bs2bt2Up.to(device)
            Bs2bt2Up.device = device
            Bs2bt2Up.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(Bs2bt2Up, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[1, mat_count] = BS_acur
            beam_acur_eval[1, mat_count] = beam_acur
            beam_norm_gain_eval[1, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[1, mat_count] = dis.detach().cpu().numpy()
            
            print('load inverse single-cascaded multi-task model!')         
            model_name = 'trained_model/Up2bt2Bs_O1_' + mat_list[mat_count] + '_velocity10ms_snr5dB_rectilinear_motion.pkl'           
            Up2bt2Bs = torch.load(model_name, map_location = device)           
            Up2bt2Bs.to(device)
            Up2bt2Bs.device = device
            Up2bt2Bs.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(Up2bt2Bs, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[2, mat_count] = BS_acur
            beam_acur_eval[2, mat_count] = beam_acur
            beam_norm_gain_eval[2, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[2, mat_count] = dis.detach().cpu().numpy()
            
            print('load vanilla multi-task model!')         
            model_name = 'trained_model/Vanilla_O1_' + mat_list[mat_count] + '_velocity10ms_snr5dB_rectilinear_motion.pkl'           
            vanilla = torch.load(model_name, map_location = device)           
            vanilla.to(device)
            vanilla.device = device
            vanilla.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(vanilla, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[3, mat_count] = BS_acur
            beam_acur_eval[3, mat_count] = beam_acur
            beam_norm_gain_eval[3, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[3, mat_count] = dis.detach().cpu().numpy()
            
            print('load single-task learning model!')         
            model_name = 'trained_model/STL_BS_selection_O1_' + mat_list[mat_count] + '_velocity10ms_snr5dB_rectilinear_motion.pkl'           
            stl_Bs = torch.load(model_name, map_location = device)           
            stl_Bs.to(device)
            stl_Bs.device = device
            stl_Bs.eval()
                        
            BS_acur = eval_Bs(stl_Bs, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
            
            model_name = 'trained_model/STL_beam_tracking_O1_' + mat_list[mat_count] + '_velocity10ms_snr5dB_rectilinear_motion.pkl'           
            stl_bt = torch.load(model_name, map_location = device)           
            stl_bt.to(device)
            stl_bt.device = device
            stl_bt.eval()
            
            beam_acur, beam_norm_gain = eval_bt(stl_bt, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
            
            model_name = 'trained_model/STL_UE_positioning_O1_' + mat_list[mat_count] + '_velocity10ms_snr5dB_rectilinear_motion.pkl'           
            stl_Up = torch.load(model_name, map_location = device)           
            stl_Up.to(device)
            stl_Up.device = device
            stl_Up.eval()
            
            dis = eval_Up(stl_Up, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[4, mat_count] = BS_acur
            beam_acur_eval[4, mat_count] = beam_acur
            beam_norm_gain_eval[4, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[4, mat_count] = dis.detach().cpu().numpy()
        
        # results visualization
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 12
        
        if args.visualize:
            
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            
            plt.figure()
            plt.plot(np.arange(0, 5), BS_acur_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), BS_acur_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), BS_acur_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), BS_acur_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 5), BS_acur_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [2]')
            plt.xlabel('Training samples')
            plt.ylabel('Prediction accuracy')
            plt.title('O1 mmWave network scenario, snr 5dB, UE velocity 10m/s, rectiinear motion')
            plt.legend(fontsize = 12)
            plt.grid(True)        
            xticks_positions = [0, 1, 2, 3, 4]
            xticks_labels = [r'$25600$', r'$51200$', r'$76800$', r'$102400$', r'$128000$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)        
            
            file_path = 'results/BS prediction accuracy with different training samples given O1 and rectilinear motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)        
            print("Have saved BS prediction accuracy with different training samples figure!")
            
            plt.figure()
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [3]')
            plt.xlabel('Training samples')
            plt.ylabel('Normalized beamforming gain')
            plt.title('O1 mmWave network scenario, snr 5dB, UE velocity 10m/s, rectiinear motion')
            plt.legend(fontsize = 12)
            plt.grid(True)       
            xticks_positions = [0, 1, 2, 3, 4]
            xticks_labels = [r'$25600$', r'$51200$', r'$76800$', r'$102400$', r'$128000$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)    
            
            file_path = 'results/Normalized beamforming gain with different training samples given O1 and rectilinear motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)      
            print("Have saved Normalized beamforming gain with different training samples figure!")
            
            plt.figure()
            plt.plot(np.arange(0, 5), dis_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), dis_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), dis_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), dis_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 5), dis_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [5]')
            plt.xlabel('Training samples')
            plt.ylabel('Average positioning error (m)')
            plt.title('O1 mmWave network scenario, snr 5dB, UE velocity 10m/s, rectiinear motion')
            plt.legend(fontsize = 12)
            plt.grid(True)       
            xticks_positions = [0, 1, 2, 3, 4]
            xticks_labels = [r'$25600$', r'$51200$', r'$76800$', r'$102400$', r'$128000$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)        
                       
            file_path = 'results/Average positioning error with different training samples given O1 and rectilinear motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)       
            print("Have saved Average positioning error with different training samples figure!")
            
    elif args.experiment_type == 'O1_velocity':
        BS_num = 4 # BS number
        velocity_list = ['5ms', '10ms', '15ms', '20ms'] # each mat corresponds to 256 data samples
        # velocity_list = ['10ms'] # each mat corresponds to 256 data samples
            
        # save results
        BS_acur_eval = np.zeros((5, len(velocity_list)))
        beam_acur_eval = np.zeros((5, len(velocity_list)))
        beam_norm_gain_eval = np.zeros((5, len(velocity_list)))
        dis_eval = np.zeros((5, len(velocity_list)))
            
        # first loop for training runnings
        for velocity_count in range(len(velocity_list)):
            print('load: ' + velocity_list[velocity_count])
            
            path_eval = 'eval_dataset/O1_v' + velocity_list[velocity_count] + '_snr5dB_rectilinearmotion'
                
            eval_loader = Dataloader(path = path_eval, batch_size = b, his_len = his_len, pre_len = pre_len, BS_num = BS_num, beam_num = beam_num, device = device)
             
            print('load dual-cascaded multi-task model!')         
            model_name = 'trained_model/Dual_Cascaded_O1_300mats_velocity' + velocity_list[velocity_count] + '_snr5dB_rectilinear_motion.pkl'           
            dual_cascaded = torch.load(model_name, map_location = device)           
            dual_cascaded.to(device)
            dual_cascaded.device = device
            dual_cascaded.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(dual_cascaded, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[0, velocity_count] = BS_acur
            beam_acur_eval[0, velocity_count] = beam_acur
            beam_norm_gain_eval[0, velocity_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[0, velocity_count] = dis.detach().cpu().numpy()
            
            print('load single-cascaded multi-task model!')         
            model_name = 'trained_model/Bs2bt2Up_O1_300mats_velocity' + velocity_list[velocity_count] + '_snr5dB_rectilinear_motion.pkl'
            Bs2bt2Up = torch.load(model_name, map_location = device)           
            Bs2bt2Up.to(device)
            Bs2bt2Up.device = device
            Bs2bt2Up.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(Bs2bt2Up, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[1, velocity_count] = BS_acur
            beam_acur_eval[1, velocity_count] = beam_acur
            beam_norm_gain_eval[1, velocity_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[1, velocity_count] = dis.detach().cpu().numpy()
            
            print('load inverse single-cascaded multi-task model!')         
            model_name = 'trained_model/Up2bt2Bs_O1_300mats_velocity' + velocity_list[velocity_count] + '_snr5dB_rectilinear_motion.pkl'      
            Up2bt2Bs = torch.load(model_name, map_location = device)           
            Up2bt2Bs.to(device)
            Up2bt2Bs.device = device
            Up2bt2Bs.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(Up2bt2Bs, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[2, velocity_count] = BS_acur
            beam_acur_eval[2, velocity_count] = beam_acur
            beam_norm_gain_eval[2, velocity_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[2, velocity_count] = dis.detach().cpu().numpy()
            
            print('load vanilla multi-task model!')         
            model_name = 'trained_model/Vanilla_O1_300mats_velocity' + velocity_list[velocity_count] + '_snr5dB_rectilinear_motion.pkl'               
            vanilla = torch.load(model_name, map_location = device)           
            vanilla.to(device)
            vanilla.device = device
            vanilla.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(vanilla, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[3, velocity_count] = BS_acur
            beam_acur_eval[3, velocity_count] = beam_acur
            beam_norm_gain_eval[3, velocity_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[3, velocity_count] = dis.detach().cpu().numpy()
            
            print('load single-task learning model!')         
            model_name = 'trained_model/STL_BS_selection_O1_300mats_velocity' + velocity_list[velocity_count] + '_snr5dB_rectilinear_motion.pkl'               
            stl_Bs = torch.load(model_name, map_location = device)           
            stl_Bs.to(device)
            stl_Bs.device = device
            stl_Bs.eval()
                        
            BS_acur = eval_Bs(stl_Bs, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
            
            model_name = 'trained_model/STL_beam_tracking_O1_300mats_velocity' + velocity_list[velocity_count] + '_snr5dB_rectilinear_motion.pkl'    
            stl_bt = torch.load(model_name, map_location = device)           
            stl_bt.to(device)
            stl_bt.device = device
            stl_bt.eval()
            
            beam_acur, beam_norm_gain = eval_bt(stl_bt, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
            
            model_name = 'trained_model/STL_UE_positioning_O1_300mats_velocity' + velocity_list[velocity_count] + '_snr5dB_rectilinear_motion.pkl'     
            stl_Up = torch.load(model_name, map_location = device)           
            stl_Up.to(device)
            stl_Up.device = device
            stl_Up.eval()
            
            dis = eval_Up(stl_Up, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[4, velocity_count] = BS_acur
            beam_acur_eval[4, velocity_count] = beam_acur
            beam_norm_gain_eval[4, velocity_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[4, velocity_count] = dis.detach().cpu().numpy()
        
        # results visualization
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 12
        
        if args.visualize:
            
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            
            plt.figure()
            plt.plot(np.arange(0, 4), BS_acur_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), BS_acur_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), BS_acur_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), BS_acur_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 4), BS_acur_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [2]')
            plt.xlabel('UE velocity v (m/s)')
            plt.ylabel('Prediction accuracy')
            plt.title('O1 mmWave network scenario, snr 5dB, varying UE velocity, rectiinear motion')
            plt.legend(fontsize = 12)
            plt.grid(True)        
            xticks_positions = [0, 1, 2, 3]
            xticks_labels = [r'$5$', r'$10$', r'$15$', r'$20$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)      
            
            file_path = 'results/BS prediction accuracy with different UE velocities given O1 and rectilinear motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)        
            print("Have saved BS prediction accuracy with different UE velocities figure!")
            
            plt.figure()
            plt.plot(np.arange(0, 4), beam_norm_gain_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), beam_norm_gain_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), beam_norm_gain_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), beam_norm_gain_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 4), beam_norm_gain_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [3]')
            plt.xlabel('UE velocity v (m/s)')
            plt.ylabel('Normalized beamforming gain')
            plt.title('O1 mmWave network scenario, snr 5dB, varying UE velocity, rectiinear motion')
            plt.legend(fontsize = 12)
            plt.grid(True)       
            xticks_positions = [0, 1, 2, 3]
            xticks_labels = [r'$5$', r'$10$', r'$15$', r'$20$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)    
            
            file_path = 'results/Normalized beamforming gain with different UE velocities given O1 and rectilinear motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)         
            print("Have saved Normalized beamforming gain with different UE velocities figure!")
            
            plt.figure()
            plt.plot(np.arange(0, 4), dis_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), dis_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), dis_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), dis_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 4), dis_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [5]')
            plt.xlabel('UE velocity v (m/s)')
            plt.ylabel('Average positioning error (m)')
            plt.title('O1 mmWave network scenario, snr 5dB, varying UE velocity, rectiinear motion')
            plt.legend(fontsize = 12)
            plt.grid(True)       
            xticks_positions = [0, 1, 2, 3]
            xticks_labels = [r'$5$', r'$10$', r'$15$', r'$20$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)    
            
            file_path = 'results/Average positioning error with different UE velocities given O1 and rectilinear motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)       
            print("Have saved Average positioning error with different UE velocities figure!")
            
    elif args.experiment_type == 'O1_snr':
        BS_num = 4 # BS number
        snr_list = ['0dB', '5dB', '10dB', '15dB'] # each mat corresponds to 256 data samples
        # snr_list = ['5dB'] # each mat corresponds to 256 data samples
            
        # save results
        BS_acur_eval = np.zeros((5, len(snr_list)))
        beam_acur_eval = np.zeros((5, len(snr_list)))
        beam_norm_gain_eval = np.zeros((5, len(snr_list)))
        dis_eval = np.zeros((5, len(snr_list)))
            
        # first loop for training runnings
        for snr_count in range(len(snr_list)):
            print('load: ' + snr_list[snr_count])
            
            path_eval = 'eval_dataset/O1_v10ms_snr' + snr_list[snr_count] + '_rectilinearmotion'
                
            eval_loader = Dataloader(path = path_eval, batch_size = b, his_len = his_len, pre_len = pre_len, BS_num = BS_num, beam_num = beam_num, device = device)
             
            print('load dual-cascaded multi-task model!')         
            model_name = 'trained_model/Dual_Cascaded_O1_300mats_velocity10ms_snr' + snr_list[snr_count] + '_rectilinear_motion.pkl'           
            dual_cascaded = torch.load(model_name, map_location = device)           
            dual_cascaded.to(device)
            dual_cascaded.device = device
            dual_cascaded.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(dual_cascaded, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[0, snr_count] = BS_acur
            beam_acur_eval[0, snr_count] = beam_acur
            beam_norm_gain_eval[0, snr_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[0, snr_count] = dis.detach().cpu().numpy()
            
            print('load single-cascaded multi-task model!')         
            model_name = 'trained_model/Bs2bt2Up_O1_300mats_velocity10ms_snr' + snr_list[snr_count] + '_rectilinear_motion.pkl'
            Bs2bt2Up = torch.load(model_name, map_location = device)           
            Bs2bt2Up.to(device)
            Bs2bt2Up.device = device
            Bs2bt2Up.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(Bs2bt2Up, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[1, snr_count] = BS_acur
            beam_acur_eval[1, snr_count] = beam_acur
            beam_norm_gain_eval[1, snr_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[1, snr_count] = dis.detach().cpu().numpy()
            
            print('load inverse single-cascaded multi-task model!')          
            model_name = 'trained_model/Up2bt2Bs_O1_300mats_velocity10ms_snr' + snr_list[snr_count] + '_rectilinear_motion.pkl'
            Up2bt2Bs = torch.load(model_name, map_location = device)           
            Up2bt2Bs.to(device)
            Up2bt2Bs.device = device
            Up2bt2Bs.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(Up2bt2Bs, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[2, snr_count] = BS_acur
            beam_acur_eval[2, snr_count] = beam_acur
            beam_norm_gain_eval[2, snr_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[2, snr_count] = dis.detach().cpu().numpy()
            
            print('load vanilla multi-task model!')         
            model_name = 'trained_model/Vanilla_O1_300mats_velocity10ms_snr' + snr_list[snr_count] + '_rectilinear_motion.pkl'            
            vanilla = torch.load(model_name, map_location = device)           
            vanilla.to(device)
            vanilla.device = device
            vanilla.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(vanilla, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[3, snr_count] = BS_acur
            beam_acur_eval[3, snr_count] = beam_acur
            beam_norm_gain_eval[3, snr_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[3, snr_count] = dis.detach().cpu().numpy()
            
            print('load single-task learning model!')         
            model_name = 'trained_model/STL_BS_selection_O1_300mats_velocity10ms_snr' + snr_list[snr_count] + '_rectilinear_motion.pkl'                         
            stl_Bs = torch.load(model_name, map_location = device)           
            stl_Bs.to(device)
            stl_Bs.device = device
            stl_Bs.eval()
                        
            BS_acur = eval_Bs(stl_Bs, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
            
            model_name = 'trained_model/STL_beam_tracking_O1_300mats_velocity10ms_snr' + snr_list[snr_count] + '_rectilinear_motion.pkl'  
            stl_bt = torch.load(model_name, map_location = device)           
            stl_bt.to(device)
            stl_bt.device = device
            stl_bt.eval()
            
            beam_acur, beam_norm_gain = eval_bt(stl_bt, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
             
            model_name = 'trained_model/STL_UE_positioning_O1_300mats_velocity10ms_snr' + snr_list[snr_count] + '_rectilinear_motion.pkl'  
            stl_Up = torch.load(model_name, map_location = device)           
            stl_Up.to(device)
            stl_Up.device = device
            stl_Up.eval()
            
            dis = eval_Up(stl_Up, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[4, snr_count] = BS_acur
            beam_acur_eval[4, snr_count] = beam_acur
            beam_norm_gain_eval[4, snr_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[4, snr_count] = dis.detach().cpu().numpy()
        
        # results visualization
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 12
        
        if args.visualize:
            
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            
            plt.figure()
            plt.plot(np.arange(0, 4), BS_acur_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), BS_acur_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), BS_acur_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), BS_acur_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 4), BS_acur_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [2]')
            plt.xlabel('Received signal SNR (dB)')
            plt.ylabel('Prediction accuracy')
            plt.title('O1 mmWave network scenario, varying snr, UE velocity 10 m/s, rectiinear motion')
            plt.legend(fontsize = 12)
            plt.grid(True)        
            xticks_positions = [0, 1, 2, 3]
            xticks_labels = [r'$0$', r'$5$', r'$10$', r'$15$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)  
            
            file_path = 'results/BS prediction accuracy with different SNRs given O1 and rectilinear motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)        
            print("Have saved BS prediction accuracy with different SNRs figure!")
            
            plt.figure()
            plt.plot(np.arange(0, 4), beam_norm_gain_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), beam_norm_gain_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), beam_norm_gain_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), beam_norm_gain_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 4), beam_norm_gain_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [3]')
            plt.xlabel('Received signal SNR (dB)')
            plt.ylabel('Normalized beamforming gain')
            plt.title('O1 mmWave network scenario, varying snr, UE velocity 10 m/s, rectiinear motion')
            plt.legend(fontsize = 12)
            plt.grid(True)       
            xticks_positions = [0, 1, 2, 3]
            xticks_labels = [r'$0$', r'$5$', r'$10$', r'$15$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)     
            
            file_path = 'results/Normalized beamforming gain with different SNRs given O1 and rectilinear motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)          
            print("Have saved Normalized beamforming gain with different SNRs figure!")
            
            plt.figure()
            plt.plot(np.arange(0, 4), dis_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), dis_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), dis_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 4), dis_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 4), dis_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [5]')
            plt.xlabel('Received signal SNR (dB)')
            plt.ylabel('Average positioning error (m)')
            plt.title('O1 mmWave network scenario, varying snr, UE velocity 10 m/s, rectiinear motion')
            plt.legend(fontsize = 12)
            plt.grid(True)       
            xticks_positions = [0, 1, 2, 3]
            xticks_labels = [r'$0$', r'$5$', r'$10$', r'$15$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)       
            
            file_path = 'results/Average positioning error with different SNRs given O1 and rectilinear motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)         
            print("Have saved Average positioning error with different SNRs figure!")
    
    elif args.experiment_type == 'O1_motion_form':
        BS_num = 4 # BS number
        path_eval = 'eval_dataset/O1_v10ms_snr5dB_spiralmotion'
        # path_eval = 'eval_dataset/tmp'
            
        eval_loader = Dataloader(path = path_eval, batch_size = b, his_len = his_len, pre_len = pre_len, BS_num = BS_num, beam_num = beam_num, device = device)
        
        
        mat_list = ['100mats', '200mats', '300mats', '400mats', '500mats'] # each mat corresponds to 256 data samples
        # mat_list = ['400mats'] # each mat corresponds to 256 data samples
            
        # save results
        BS_acur_eval = np.zeros((5, len(mat_list)))
        beam_acur_eval = np.zeros((5, len(mat_list)))
        beam_norm_gain_eval = np.zeros((5, len(mat_list)))
        dis_eval = np.zeros((5, len(mat_list)))
        
    
        # first loop for training runnings
        for mat_count in range(len(mat_list)):
            print('load: ' + mat_list[mat_count])
             
            print('load dual-cascaded multi-task model!')         
            model_name = 'trained_model/Dual_Cascaded_O1_' + mat_list[mat_count] + '_velocity10ms_snr5dB_spiral_motion.pkl'           
            dual_cascaded = torch.load(model_name, map_location = device)           
            dual_cascaded.to(device)
            dual_cascaded.device = device
            dual_cascaded.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(dual_cascaded, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[0, mat_count] = BS_acur
            beam_acur_eval[0, mat_count] = beam_acur
            beam_norm_gain_eval[0, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[0, mat_count] = dis.detach().cpu().numpy()
            
            print('load single-cascaded multi-task model!')         
            model_name = 'trained_model/Bs2bt2Up_O1_' + mat_list[mat_count] + '_velocity10ms_snr5dB_spiral_motion.pkl'           
            Bs2bt2Up = torch.load(model_name, map_location = device)           
            Bs2bt2Up.to(device)
            Bs2bt2Up.device = device
            Bs2bt2Up.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(Bs2bt2Up, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[1, mat_count] = BS_acur
            beam_acur_eval[1, mat_count] = beam_acur
            beam_norm_gain_eval[1, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[1, mat_count] = dis.detach().cpu().numpy()
            
            print('load inverse single-cascaded multi-task model!')         
            model_name = 'trained_model/Up2bt2Bs_O1_' + mat_list[mat_count] + '_velocity10ms_snr5dB_spiral_motion.pkl'           
            Up2bt2Bs = torch.load(model_name, map_location = device)           
            Up2bt2Bs.to(device)
            Up2bt2Bs.device = device
            Up2bt2Bs.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(Up2bt2Bs, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[2, mat_count] = BS_acur
            beam_acur_eval[2, mat_count] = beam_acur
            beam_norm_gain_eval[2, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[2, mat_count] = dis.detach().cpu().numpy()
            
            print('load vanilla multi-task model!')         
            model_name = 'trained_model/Vanilla_O1_' + mat_list[mat_count] + '_velocity10ms_snr5dB_spiral_motion.pkl'           
            vanilla = torch.load(model_name, map_location = device)           
            vanilla.to(device)
            vanilla.device = device
            vanilla.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(vanilla, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[3, mat_count] = BS_acur
            beam_acur_eval[3, mat_count] = beam_acur
            beam_norm_gain_eval[3, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[3, mat_count] = dis.detach().cpu().numpy()
            
            print('load single-task learning model!')         
            model_name = 'trained_model/STL_BS_selection_O1_' + mat_list[mat_count] + '_velocity10ms_snr5dB_spiral_motion.pkl'           
            stl_Bs = torch.load(model_name, map_location = device)           
            stl_Bs.to(device)
            stl_Bs.device = device
            stl_Bs.eval()
                        
            BS_acur = eval_Bs(stl_Bs, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
            
            model_name = 'trained_model/STL_beam_tracking_O1_' + mat_list[mat_count] + '_velocity10ms_snr5dB_spiral_motion.pkl'           
            stl_bt = torch.load(model_name, map_location = device)           
            stl_bt.to(device)
            stl_bt.device = device
            stl_bt.eval()
            
            beam_acur, beam_norm_gain = eval_bt(stl_bt, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
            
            model_name = 'trained_model/STL_UE_positioning_O1_' + mat_list[mat_count] + '_velocity10ms_snr5dB_spiral_motion.pkl'           
            stl_Up = torch.load(model_name, map_location = device)           
            stl_Up.to(device)
            stl_Up.device = device
            stl_Up.eval()
            
            dis = eval_Up(stl_Up, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[4, mat_count] = BS_acur
            beam_acur_eval[4, mat_count] = beam_acur
            beam_norm_gain_eval[4, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[4, mat_count] = dis.detach().cpu().numpy()
        
        # results visualization
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 12
        
        if args.visualize:
            
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            
            plt.figure()
            plt.plot(np.arange(0, 5), BS_acur_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), BS_acur_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), BS_acur_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), BS_acur_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 5), BS_acur_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [2]')
            plt.xlabel('Training samples')
            plt.ylabel('Prediction accuracy')
            plt.title('O1 mmWave network scenario, snr 5dB, UE velocity 10m/s, spiral motion')
            plt.legend(fontsize = 12)
            plt.grid(True)        
            xticks_positions = [0, 1, 2, 3, 4]
            xticks_labels = [r'$25600$', r'$51200$', r'$76800$', r'$102400$', r'$128000$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)        
            
            file_path = 'results/BS prediction accuracy with different training samples given O1 and spiral motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)      
            print("Have saved BS prediction accuracy with different training samples figure!")
            
            plt.figure()
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [3]')
            plt.xlabel('Training samples')
            plt.ylabel('Normalized beamforming gain')
            plt.title('O1 mmWave network scenario, snr 5dB, UE velocity 10m/s, spiral motion')
            plt.legend(fontsize = 12)
            plt.grid(True)       
            xticks_positions = [0, 1, 2, 3, 4]
            xticks_labels = [r'$25600$', r'$51200$', r'$76800$', r'$102400$', r'$128000$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)  
            
            file_path = 'results/Normalized beamforming gain with different training samples given O1 and spiral motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)          
            print("Have saved Normalized beamforming gain with different training samples figure!")
            
            plt.figure()
            plt.plot(np.arange(0, 5), dis_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), dis_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), dis_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), dis_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 5), dis_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [5]')
            plt.xlabel('Training samples')
            plt.ylabel('Average positioning error (m)')
            plt.title('O1 mmWave network scenario, snr 5dB, UE velocity 10m/s, spiral motion')
            plt.legend(fontsize = 12)
            plt.grid(True)       
            xticks_positions = [0, 1, 2, 3, 4]
            xticks_labels = [r'$25600$', r'$51200$', r'$76800$', r'$102400$', r'$128000$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)        
            
            file_path = 'results/Average positioning error with different training samples given O1 and spiral motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)        
            print("Have saved Average positioning error with different training samples figure!")        
    
    elif args.experiment_type == 'Outdoor_Blockage_trainingsamples':
        BS_num = 3 # BS number
        
        path_eval = 'eval_dataset/O1B_v10ms_snr5dB_rectilinearmotion'
        # path_eval = 'eval_dataset/tmp'
            
        eval_loader = Dataloader(path = path_eval, batch_size = b, his_len = his_len, pre_len = pre_len, BS_num = BS_num, beam_num = beam_num, device = device)
        
        mat_list = ['100mats', '200mats', '300mats', '400mats', '500mats'] # each mat corresponds to 256 data samples
        # mat_list = ['400mats'] # each mat corresponds to 256 data samples
            
        # save results
        BS_acur_eval = np.zeros((5, len(mat_list)))
        beam_acur_eval = np.zeros((5, len(mat_list)))
        beam_norm_gain_eval = np.zeros((5, len(mat_list)))
        dis_eval = np.zeros((5, len(mat_list)))
        
    
        # first loop for training runnings
        for mat_count in range(len(mat_list)):
            print('load: ' + mat_list[mat_count])
             
            print('load dual-cascaded multi-task model!')         
            model_name = 'trained_model/Dual_Cascaded_O1_Blockage_' + mat_list[mat_count] + '_velocity10ms_snr5dB_rectilinear_motion.pkl'           
            dual_cascaded = torch.load(model_name, map_location = device)           
            dual_cascaded.to(device)
            dual_cascaded.device = device
            dual_cascaded.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(dual_cascaded, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[0, mat_count] = BS_acur
            beam_acur_eval[0, mat_count] = beam_acur
            beam_norm_gain_eval[0, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[0, mat_count] = dis.detach().cpu().numpy()
            
            print('load single-cascaded multi-task model!')         
            model_name = 'trained_model/Bs2bt2Up_O1_Blockage_' + mat_list[mat_count] + '_velocity10ms_snr5dB_rectilinear_motion.pkl'           
            Bs2bt2Up = torch.load(model_name, map_location = device)           
            Bs2bt2Up.to(device)
            Bs2bt2Up.device = device
            Bs2bt2Up.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(Bs2bt2Up, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[1, mat_count] = BS_acur
            beam_acur_eval[1, mat_count] = beam_acur
            beam_norm_gain_eval[1, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[1, mat_count] = dis.detach().cpu().numpy()
            
            print('load inverse single-cascaded multi-task model!')         
            model_name = 'trained_model/Up2bt2Bs_O1_Blockage_' + mat_list[mat_count] + '_velocity10ms_snr5dB_rectilinear_motion.pkl'           
            Up2bt2Bs = torch.load(model_name, map_location = device)           
            Up2bt2Bs.to(device)
            Up2bt2Bs.device = device
            Up2bt2Bs.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(Up2bt2Bs, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[2, mat_count] = BS_acur
            beam_acur_eval[2, mat_count] = beam_acur
            beam_norm_gain_eval[2, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[2, mat_count] = dis.detach().cpu().numpy()
            
            print('load vanilla multi-task model!')         
            model_name = 'trained_model/Vanilla_O1_Blockage_' + mat_list[mat_count] + '_velocity10ms_snr5dB_rectilinear_motion.pkl'           
            vanilla = torch.load(model_name, map_location = device)           
            vanilla.to(device)
            vanilla.device = device
            vanilla.eval()
                        
            BS_acur, beam_acur, beam_norm_gain, dis = eval_Bs_bt_Up(vanilla, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[3, mat_count] = BS_acur
            beam_acur_eval[3, mat_count] = beam_acur
            beam_norm_gain_eval[3, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[3, mat_count] = dis.detach().cpu().numpy()
            
            print('load single-task learning model!')         
            model_name = 'trained_model/STL_BS_selection_O1_Blockage_' + mat_list[mat_count] + '_velocity10ms_snr5dB_rectilinear_motion.pkl'           
            stl_Bs = torch.load(model_name, map_location = device)           
            stl_Bs.to(device)
            stl_Bs.device = device
            stl_Bs.eval()
                        
            BS_acur = eval_Bs(stl_Bs, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
            
            model_name = 'trained_model/STL_beam_tracking_O1_Blockage_' + mat_list[mat_count] + '_velocity10ms_snr5dB_rectilinear_motion.pkl'           
            stl_bt = torch.load(model_name, map_location = device)           
            stl_bt.to(device)
            stl_bt.device = device
            stl_bt.eval()
            
            beam_acur, beam_norm_gain = eval_bt(stl_bt, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
            
            model_name = 'trained_model/STL_UE_positioning_O1_Blockage_' + mat_list[mat_count] + '_velocity10ms_snr5dB_rectilinear_motion.pkl'           
            stl_Up = torch.load(model_name, map_location = device)           
            stl_Up.to(device)
            stl_Up.device = device
            stl_Up.eval()
            
            dis = eval_Up(stl_Up, eval_loader, b, his_len, pre_len, BS_num, beam_num, device)
                    
            BS_acur_eval[4, mat_count] = BS_acur
            beam_acur_eval[4, mat_count] = beam_acur
            beam_norm_gain_eval[4, mat_count] = beam_norm_gain.detach().cpu().numpy()
            dis_eval[4, mat_count] = dis.detach().cpu().numpy()
        
        # results visualization
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 12
        
        if args.visualize:
            
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            
            plt.figure()
            plt.plot(np.arange(0, 5), BS_acur_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), BS_acur_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), BS_acur_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), BS_acur_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 5), BS_acur_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [2]')
            plt.xlabel('Training samples')
            plt.ylabel('Prediction accuracy')
            plt.title('O1 Blockage mmWave network scenario, snr 5dB, UE velocity 10m/s, rectiinear motion')
            plt.legend(fontsize = 12)
            plt.grid(True)        
            xticks_positions = [0, 1, 2, 3, 4]
            xticks_labels = [r'$25600$', r'$51200$', r'$76800$', r'$102400$', r'$128000$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)  
            
            file_path = 'results/BS prediction accuracy with different training samples given O1 Blockage and rectilinear motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)      
            print("Have saved BS prediction accuracy with different training samples figure!")
            
            plt.figure()
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 5), beam_norm_gain_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [3]')
            plt.xlabel('Training samples')
            plt.ylabel('Normalized beamforming gain')
            plt.title('O1 Blockage mmWave network scenario, snr 5dB, UE velocity 10m/s, rectiinear motion')
            plt.legend(fontsize = 12)
            plt.grid(True)       
            xticks_positions = [0, 1, 2, 3, 4]
            xticks_labels = [r'$25600$', r'$51200$', r'$76800$', r'$102400$', r'$128000$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)
            
            file_path = 'results/Normalized beamforming gain with different training samples given O1 Blockage and rectilinear motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)       
            print("Have saved Normalized beamforming gain with different training samples figure!")
            
            plt.figure()
            plt.plot(np.arange(0, 5), dis_eval[0, :], '^-', linewidth = 1.5, color = (1, 0, 0), label = 'Proposed dual-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), dis_eval[1, :], '*-', linewidth = 1.5, color = (0, 0.4470, 0.7410), label = 'Single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), dis_eval[2, :], 'x-', linewidth = 1.5, color = (0.4660, 0.6740, 0.1880), label = 'Inverse single-cascaded multi-task learning')
            plt.plot(np.arange(0, 5), dis_eval[3, :], 'd-', linewidth = 1.5, color = (0.4940, 0.1840, 0.5560), label = 'Vanilla multi-task learning [9]')
            plt.plot(np.arange(0, 5), dis_eval[4, :], 's-', linewidth = 1.5, color = (0.8500, 0.3250, 0.0980), label = 'Single-task learning [5]')
            plt.xlabel('Training samples')
            plt.ylabel('Average positioning error (m)')
            plt.title('O1 Blockage mmWave network scenario, snr 5dB, UE velocity 10m/s, rectiinear motion')
            plt.legend(fontsize = 12)
            plt.grid(True)       
            xticks_positions = [0, 1, 2, 3, 4]
            xticks_labels = [r'$25600$', r'$51200$', r'$76800$', r'$102400$', r'$128000$']
            plt.xticks(ticks=xticks_positions, labels = xticks_labels, fontsize = 12)      
            
            file_path = 'results/Average positioning error with different training samples given O1 Blockage and rectilinear motion.png'
            if os.path.exists(file_path):
                os.remove(file_path)            
            plt.savefig(file_path)        
            print("Have saved Average positioning error with different training samples figure!")
            
        else:
            raise NotImplementedError
        

if __name__ == '__main__':
    main()