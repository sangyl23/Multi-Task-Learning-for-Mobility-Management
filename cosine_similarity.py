import torch.optim as optim
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from dataloader_MTL import Dataloader
from model_MTL import Dual_Cascaded
import sys
import time
import argparse
import logging

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("logfile.txt", mode = 'w'),
    logging.StreamHandler()
])
logger = logging.getLogger()

parser = argparse.ArgumentParser()

parser.add_argument('--grad_calculate', type = eval, default = True)

args = parser.parse_args()

# model_evaluation
def eval(model, loader, b, his_len, pre_len, BS_num, beam_num, criterion_Bs, criterion_bt, criterion_Up, omega_Bs, omega_bt, omega_Up, device):
    # reset dataloader
    loader.reset()
    # judge whether dataset is finished
    done = False
    # running loss
    running_loss = 0.
    running_loss_Bs = 0.
    running_loss_bt = 0.
    running_loss_Up = 0.
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

            loss_Bs = criterion_Bs(out_BS_label, BS_label_pre)
            loss_bt = criterion_bt(out_beam_label, beam_label_pre)
            loss_Up = criterion_Up(out_loc, UE_loc_pre)
            
            loss = omega_Bs * loss_Bs + omega_bt * loss_bt + omega_Up * loss_Up
            
            running_loss += loss.item()
            running_loss_Bs += loss_Bs.item()
            running_loss_bt += loss_bt.item()
            running_loss_Up += loss_Up.item()
            BS_top1_acc += (torch.sum(torch.argmax(out_BS_label, dim = 1) == BS_label_pre) / b).item()
            beam_top1_acc += (torch.sum(torch.argmax(out_beam_label, dim = 1) == beam_label_pre) / b).item()      
            
            gt_beampower_idx = beam_label_pre.unsqueeze(dim = 1) # (b, 1)
            pre_beampower_idx = torch.argmax(out_beam_label, dim = 1)
            pre_beampower_idx = pre_beampower_idx.unsqueeze(dim = 1) # (b, 1)
            beam_norm_gain += torch.mean(torch.squeeze(torch.gather(beam_power_pre, 1, pre_beampower_idx), dim = 1) / torch.squeeze(torch.gather(beam_power_pre, 1, gt_beampower_idx), dim = 1))
            
            dis_l2 += torch.mean(torch.sqrt((out_loc[:, 0] - UE_loc_pre[:, 0]) ** 2 + (out_loc[:, 1] - UE_loc_pre[:, 1]) ** 2))
            
    losses = running_loss / batch_num
    losses_Bs = running_loss_Bs / batch_num
    losses_bt = running_loss_bt / batch_num
    losses_Up = running_loss_Up / batch_num
    BS_acur = BS_top1_acc / batch_num
    beam_acur = beam_top1_acc / batch_num
    beam_norm_gain = beam_norm_gain / batch_num
    dis = dis_l2 / batch_num

    # print results
    print("Eval loss: %.3f" % (losses))
    print("Eval BS-selection loss: %.3f" % (losses_Bs))
    print("Eval beam-tracking loss: %.3f" % (losses_bt))
    print("Eval UE-positioning loss: %.3f" % (losses_Up))
    print("Eval BS-selection accuracy: %.3f" % (BS_acur))
    print("Eval beam-tracking accuracy: %.3f" % (beam_acur))
    print("Eval beam-tracking normalized gain: %.3f" % (beam_norm_gain))
    print("Eval distance error: %.3f" % (dis))

    return losses, losses_Bs, losses_bt, losses_Up, BS_acur, beam_acur, beam_norm_gain, dis


# main function for model training and evaluation
# output: accuracy, losses and normalized beamforming gain
def main(training_time = 3, epoch_num = 100, batch_size = 64, lr = 1e-3, minlr = 1e-9):
    version_name = 'MTL_BS1-10-14-17_Row1400-1650_v10ms_snr5dB' + '_lr' + str(lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print basic information
    print(version_name)
    print('device:%s' % device)
    print('batch_size:%d' % batch_size)
    print('lr and minlr:(%e,%e)'%(lr,minlr))
    
    for arg in vars(args):
        logger.info(f'{arg} = {getattr(args, arg)}')
    
    # system parameters
    b = batch_size
    his_len = 9 # length of historical sequence
    pre_len = 1 # length of predicted sequence
    BS_num = 4 # BS number
    beam_num = 32 # beam number
    BS_dim = 'feature_map' # how to place BS number dim for cnn feature extraction

    # training set and validation set
    path_train = 'eval_dataset/O1_v10ms_snr5dB_rectilinearmotion'
    path_eval = 'eval_dataset/O1_v10ms_snr5dB_rectilinearmotion'
    loader = Dataloader(path = path_train, batch_size = b, his_len = his_len, pre_len = pre_len, BS_num = BS_num, beam_num = beam_num, device = device)
    eval_loader = Dataloader(path = path_eval, batch_size = b, his_len = his_len, pre_len = pre_len, BS_num = BS_num, beam_num = beam_num, device = device)

    # loss function
    criterion_Bs = nn.CrossEntropyLoss()
    criterion_bt = nn.CrossEntropyLoss()
    criterion_Up = nn.MSELoss()

    # save results
    loss_eval = np.zeros((training_time, 4, epoch_num))
    loss_train = np.zeros((training_time, 4, epoch_num))
    BS_acur_train = np.zeros((training_time, epoch_num))
    BS_acur_eval = np.zeros((training_time, epoch_num))
    beam_acur_train = np.zeros((training_time, epoch_num))
    beam_acur_eval = np.zeros((training_time, epoch_num))
    beam_norm_gain_train = np.zeros((training_time, epoch_num))
    beam_norm_gain_eval = np.zeros((training_time, epoch_num))
    dis_train = np.zeros((training_time, epoch_num))
    dis_eval = np.zeros((training_time, epoch_num))
    now_lr = np.zeros((training_time, epoch_num))
    omega_list = np.zeros((training_time, 3, epoch_num))
    training_duration_list = np.zeros((training_time, epoch_num))
    task_correlations_measure_list = np.zeros((training_time, epoch_num))

    # first loop for training runnings
    for tt in range(training_time):
        print('Train %d times' % (tt))

        # model initialization
        model = Dual_Cascaded(his_len = his_len, 
                      pre_len = pre_len, 
                      BS_num = BS_num, 
                      beam_num = beam_num, 
                      cnn_feature_num = 64, 
                      lstm_feature_num = 512, 
                      cascaded_lstm_dropout = 0.5, 
                      BS_dim = BS_dim, 
                      device = device)
        model.to(device)
        # Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999))
        # learning rate adaptive decay
        lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, patience = 2,
                                                              verbose = True, threshold = 0.0001,
                                                              threshold_mode = 'rel', cooldown = 0, min_lr = minlr,
                                                              eps = 1e-08)
        min_losses = 1e10
        
        # second loop for training times
        for e in range(epoch_num):
            start = time.time()
            print('Train %d epoch' % (e))
            # reset the dataloader
            loader.reset()
            eval_loader.reset()
            # judge whether data loading is done
            done = False
            # running loss
            running_loss = 0.
            running_loss_Bs = 0.
            running_loss_bt = 0.
            running_loss_Up = 0.
            # top1_acc
            BS_top1_acc = 0.
            beam_top1_acc = 0.
            beam_norm_gain = 0.
            # l2 norm distance
            dis_l2 = 0.
            # count training time
            training_duration = 0.
            # measure task correlations
            task_correlations_measure = 0.
            
            omega_Bs = torch.tensor([1.], device = device)
            omega_bt = torch.tensor([1.], device = device)
            omega_Up = torch.tensor([1.], device = device)

            # count batch number
            batch_num = 0
            
            now_lr[tt, e] = optimizer.state_dict()['param_groups'][0]['lr']
            print('This epoch lr is %.10f' % (now_lr[tt, e]))
            
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
                training_start = time.time()
                optimizer.zero_grad()
                out_BS_label, out_beam_label, out_loc = model(channel_his) 

                loss_Bs = criterion_Bs(out_BS_label, BS_label_pre)
                loss_bt = criterion_bt(out_beam_label, beam_label_pre)
                loss_Up = criterion_Up(out_loc, UE_loc_pre)
                
                # print("Training BS-selection loss: %.3f" % (loss_Bs))
                # print("Training beam-tracking loss: %.3f" % (loss_bt))
                # print("Training UE-positioning loss: %.3f" % (loss_Up))
                
                #######################################################
                # cosine similarity calculation between multi-task gradients
                if args.grad_calculate:
                    weights = torch.cat([omega_Bs, omega_bt, omega_Up]) # (3,)
                    loss = torch.stack([loss_Bs, loss_bt, loss_Up]) # (3,)                    
                    grad_list = []
                    for i in range(len(loss)):
                        loss_i = weights[i] * loss[i]
                        loss_i.backward(retain_graph = True)
                        grad_list.append(model.Encoder[-1].conv.weight.grad.clone())
                        optimizer.zero_grad()
                    
                    with torch.no_grad():                            
                        vec1 = grad_list[0] + grad_list[1] # (512, 64, 3)
                        vec2 = grad_list[2] # (512, 64, 3)
                        
                        vec1 = vec1.reshape(-1, vec1.shape[2]) # (512 * 64, 3)
                        vec2 = vec2.reshape(-1, vec2.shape[2]) # (512 * 64, 3)
                        
                        vec2_norms = torch.linalg.norm(vec2, dim = 1) # (512 * 64,)
                        
                        # get max 1
                        _, top_indices = torch.topk(vec2_norms, 1, dim = 0) # (1,), (1,)
                        
                        vec1_top = vec1[top_indices, :] # (1, 3)
                        vec2_top = vec2[top_indices, :] # (1, 3)
                        
                        task_correlations_measure += torch.mean(torch.sum(vec1_top * vec2_top, dim = 1) / torch.linalg.norm(vec1_top, dim = 1) / torch.linalg.norm(vec2_top, dim = 1), dim = 0)
                #######################################################
                
                loss = omega_Bs * loss_Bs + omega_bt * loss_bt + omega_Up * loss_Up                
                # gradient back propagation
                loss.backward()               
                # parameter optimization
                optimizer.step()
                training_duration += time.time() - training_start
                
                running_loss += loss.item()
                running_loss_Bs += loss_Bs.item()
                running_loss_bt += loss_bt.item()
                running_loss_Up += loss_Up.item()
                BS_top1_acc += (torch.sum(torch.argmax(out_BS_label, dim = 1) == BS_label_pre) / b).item()
                beam_top1_acc += (torch.sum(torch.argmax(out_beam_label, dim = 1) == beam_label_pre) / b).item()      
                
                gt_beampower_idx = beam_label_pre.unsqueeze(dim = 1) # (b, 1)
                pre_beampower_idx = torch.argmax(out_beam_label, dim = 1)
                pre_beampower_idx = pre_beampower_idx.unsqueeze(dim = 1) # (b, 1)
                beam_norm_gain += torch.mean(torch.squeeze(torch.gather(beam_power_pre, 1, pre_beampower_idx), dim = 1) / torch.squeeze(torch.gather(beam_power_pre, 1, gt_beampower_idx), dim = 1))
                
                dis_l2 += torch.mean(torch.sqrt((out_loc[:, 0] - UE_loc_pre[:, 0]) ** 2 + (out_loc[:, 1] - UE_loc_pre[:, 1]) ** 2))
                
            losses = running_loss / batch_num
            losses_Bs = running_loss_Bs / batch_num
            losses_bt = running_loss_bt / batch_num
            losses_Up = running_loss_Up / batch_num
            BS_acur = BS_top1_acc / batch_num
            beam_acur = beam_top1_acc / batch_num
            beam_norm_gain = beam_norm_gain / batch_num
            dis = dis_l2 / batch_num   
            
            # print results
            print("Training loss: %.3f" % (losses))
            print("Training BS-selection loss: %.3f" % (losses_Bs))
            print("Training beam-tracking loss: %.3f" % (losses_bt))
            print("Training UE-positioning loss: %.3f" % (losses_Up))
            print("Training BS-selection accuracy: %.3f" % (BS_acur))
            print("Training beam-tracking accuracy: %.3f" % (beam_acur))
            print("Training beam-tracking normalized gain: %.3f" % (beam_norm_gain))
            print("Training distance error: %.3f" % (dis))
            print("Training duration: %.3f" % (training_duration))
            
            if args.grad_calculate:
                task_correlations_measure = task_correlations_measure / batch_num
                print("Task correlations measure: %.5f" % (task_correlations_measure))
            
            loss_train[tt, 0, e] = losses
            loss_train[tt, 1, e] = losses_Bs
            loss_train[tt, 2, e] = losses_bt
            loss_train[tt, 3, e] = losses_Up
            BS_acur_train[tt, e] = BS_acur
            beam_acur_train[tt, e] = beam_acur
            beam_norm_gain_train[tt, e] = beam_norm_gain
            dis_train[tt, e] = dis
            omega_list[tt, 0, e] = omega_Bs
            omega_list[tt, 1, e] = omega_bt
            omega_list[tt, 2, e] = omega_Up
            training_duration_list[tt, e] = training_duration
            task_correlations_measure_list[tt, e] = task_correlations_measure
            # eval mode, where dropout is off
            model.eval()
       
            losses, losses_Bs, losses_bt, losses_Up, BS_acur, beam_acur, beam_norm_gain, dis = eval(model, eval_loader, b, his_len, pre_len, BS_num, beam_num, 
                                                                                                    criterion_Bs, criterion_bt, criterion_Up, omega_Bs, omega_bt, omega_Up, device)
            loss_eval[tt, 0, e] = losses
            loss_eval[tt, 1, e] = losses_Bs
            loss_eval[tt, 2, e] = losses_bt
            loss_eval[tt, 3, e] = losses_Up
            BS_acur_eval[tt, e] = BS_acur
            beam_acur_eval[tt, e] = beam_acur
            beam_norm_gain_eval[tt, e] = beam_norm_gain
            dis_eval[tt, e] = dis

            lr_decay.step(losses)
            # train mode, where dropout is on
            model.train()

            # save results into mat file
            mat_name = version_name + '.mat'
            sio.savemat(mat_name, {'BS_acur_train': BS_acur_train,
                                   'BS_acur_eval': BS_acur_eval,
                                   'beam_acur_train': beam_acur_train,
                                   'beam_acur_eval': beam_acur_eval,
                                   'beam_norm_gain_train': beam_norm_gain_train,
                                   'beam_norm_gain_eval': beam_norm_gain_eval,
                                   'dis_train': dis_train,
                                   'dis_eval': dis_eval,
                                   'loss_train': loss_train, 
                                   'loss_eval': loss_eval,
                                   'now_lr': now_lr,
                                   'omega_list': omega_list,
                                   'training_duration_list': training_duration_list,
                                   'task_correlations_measure_list': task_correlations_measure_list
                                   })
            
            if losses < min_losses:
                min_losses = losses
                model_name = version_name + '_tt' + str(tt) + '.pkl'
                torch.save(model, model_name)
            
            print('This epoch takes %.1f s' % (time.time() - start))

if __name__ == '__main__':
    main()