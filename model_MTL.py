import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class CB(nn.Module):
    def __init__(self, c_i = 64, c_o = 64, k = 3, s = 1, p = 1):
        super(CB, self).__init__()
        
        self.conv = nn.Conv1d(in_channels = c_i, out_channels = c_o,
                               kernel_size = k, stride = s, padding = p)
        self.bn = nn.BatchNorm1d(c_o)
        self.activation = nn.ReLU()        

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
class Bs(nn.Module):

    def __init__(self, his_len = 9, pre_len = 1, BS_num = 4, beam_num = 64, BS_dim = 'feature_map', device = 'cpu'):
        super(Bs, self).__init__()
        
        self.his_len = his_len
        self.BS_num = BS_num
        self.beam_num = beam_num
        self.BS_dim = BS_dim
        self.device = device
        
        # Encoder
        self.Encoder = []       
        if BS_dim == 'feature_channel':
            print('you choose feature channel for BS dim!') 
            self.bn0 = nn.BatchNorm1d(2 * BS_num)
            self.Encoder.append(CB(c_i = 2 * BS_num, c_o = 64, k = 3, s = 3, p = 1))
        else:   
            print('you choose feature map for BS dim!')       
            self.bn0 = nn.BatchNorm1d(2)
            self.Encoder.append(CB(c_i = 2, c_o = 64, k = 3, s = 3, p = 1))
        self.Encoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1))
        self.Encoder.append(CB(c_i = 64, c_o = 512, k = 3, s = 3, p = 1))   
        # self.Encoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1)) 
        self.Encoder = nn.Sequential(*self.Encoder)
       
        # LSTM layer for extracting temporal correlations
        self.en_lstm = nn.LSTMCell(input_size = 512, hidden_size = 512)
                
        # Decoder
        self.Decoder = []        
        self.Decoder.append(CB(c_i = 16, c_o = 64, k = 3, s = 3, p = 1))
        self.Decoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1))
        self.Decoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1))  
        # self.Decoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1))              
        self.Decoder = nn.Sequential(*self.Decoder)
        
        # FC for output
        self.out_fc = nn.Linear(64, BS_num)

        
    def forward(self, x):
        
        # x: size (b, 2, his_len, BS_num, beam_num)
        _, _, his_len, BS_num, beam_num = x.shape
        
        if self.BS_dim == 'feature_channel':
            x = x.permute(0, 2, 1, 3, 4) # (b, his_len, 2, BS_num, beam_num)
            x = x.reshape(-1, 2, BS_num, beam_num) # (b * his_len, 2, BS_num, beam_num)
            x = x.reshape(x.shape[0], -1, beam_num) # (b * his_len, 2 * BS_num, beam_num)
        else:
            x = x.permute(0, 2, 1, 3, 4) # (b, his_len, 2, BS_num, beam_num)
            x = x.reshape(-1, 2, BS_num, beam_num) # (b * his_len, 2, BS_num, beam_num)
            x = x.reshape(x.shape[0], 2, -1) # (b * his_len, 2, BS_num * beam_num), note that beams for the same BS should be adjacent
                    
        # BN
        x = self.bn0(x)
        
        # Encoder
        x = self.Encoder(x)
        
        # Pooling
        x = nn.AvgPool1d(kernel_size = (x.shape[2]))(x) # (b * his_len, c_o, 1)
        x = torch.squeeze(x) # (b * his_len, c_o)
        x = x.reshape(-1, self.his_len, x.shape[1]) # (b, his_len, c_o)     
        x = x.permute(1, 0, 2) # (his_len, b, c_o)
        
        h0_Bs = torch.zeros((x.shape[1], 512), device = self.device, requires_grad = True)
        c0_Bs = torch.zeros((x.shape[1], 512), device = self.device, requires_grad = True)
        
        # LSTM learns temporal correlations
        for t in range(his_len):
          inputs_Bs = x[t, :, :] # (b, c_o)
          h0_Bs, c0_Bs = self.en_lstm(inputs_Bs, [h0_Bs, c0_Bs]) 
                  
        h = h0_Bs.reshape(h0_Bs.shape[0], -1, beam_num) # (b, *, beam_num)
        
        # Decoder
        h = self.Decoder(h)
        
        # Pooling
        h = nn.AvgPool1d(kernel_size = (h.shape[2]))(h) # (b, c_i, 1)
        h = torch.squeeze(h) # (b, c_i)
        
        # Output
        y = self.out_fc(h)
                
        # (b, beam_num)
        
        return y

class bt(nn.Module):

    def __init__(self, his_len = 9, pre_len = 1, BS_num = 4, beam_num = 64, BS_dim = 'feature_map', device = 'cpu'):
        super(bt, self).__init__()
        
        self.his_len = his_len
        self.BS_num = BS_num
        self.beam_num = beam_num
        self.BS_dim = BS_dim
        self.device = device
        
        # Encoder
        self.Encoder = []       
        if BS_dim == 'feature_channel':
            print('you choose feature channel for BS dim!') 
            self.bn0 = nn.BatchNorm1d(2 * BS_num)
            self.Encoder.append(CB(c_i = 2 * BS_num, c_o = 64, k = 3, s = 3, p = 1))
        else:   
            print('you choose feature map for BS dim!')       
            self.bn0 = nn.BatchNorm1d(2)
            self.Encoder.append(CB(c_i = 2, c_o = 64, k = 3, s = 3, p = 1))
        self.Encoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1))
        self.Encoder.append(CB(c_i = 64, c_o = 512, k = 3, s = 3, p = 1))   
        # self.Encoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1)) 
        self.Encoder = nn.Sequential(*self.Encoder)
       
        # LSTM layer for extracting temporal correlations
        self.en_lstm = nn.LSTMCell(input_size = 512, hidden_size = 512)
                
        # Decoder
        self.Decoder = []        
        self.Decoder.append(CB(c_i = 16, c_o = 64, k = 3, s = 3, p = 1))
        self.Decoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1))
        self.Decoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1))  
        # self.Decoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1))              
        self.Decoder = nn.Sequential(*self.Decoder)
        
        # FC for output
        self.out_fc = nn.Linear(64, BS_num * beam_num)

        
    def forward(self, x):
        
        # x: size (b, 2, his_len, BS_num, beam_num)
        _, _, his_len, BS_num, beam_num = x.shape
        
        if self.BS_dim == 'feature_channel':
            x = x.permute(0, 2, 1, 3, 4) # (b, his_len, 2, BS_num, beam_num)
            x = x.reshape(-1, 2, BS_num, beam_num) # (b * his_len, 2, BS_num, beam_num)
            x = x.reshape(x.shape[0], -1, beam_num) # (b * his_len, 2 * BS_num, beam_num)
        else:
            x = x.permute(0, 2, 1, 3, 4) # (b, his_len, 2, BS_num, beam_num)
            x = x.reshape(-1, 2, BS_num, beam_num) # (b * his_len, 2, BS_num, beam_num)
            x = x.reshape(x.shape[0], 2, -1) # (b * his_len, 2, BS_num * beam_num), note that beams for the same BS should be adjacent
                    
        # BN
        x = self.bn0(x)
        
        # Encoder
        x = self.Encoder(x)
        
        # Pooling
        x = nn.AvgPool1d(kernel_size = (x.shape[2]))(x) # (b * his_len, c_o, 1)
        x = torch.squeeze(x) # (b * his_len, c_o)
        x = x.reshape(-1, self.his_len, x.shape[1]) # (b, his_len, c_o)     
        x = x.permute(1, 0, 2) # (his_len, b, c_o)
        
        h0_bt = torch.zeros((x.shape[1], 512), device = self.device, requires_grad = True)
        c0_bt = torch.zeros((x.shape[1], 512), device = self.device, requires_grad = True)
        
        # LSTM learns temporal correlations
        for t in range(his_len):
          inputs_bt = x[t, :, :] # (b, c_o)
          h0_bt, c0_bt = self.en_lstm(inputs_bt, [h0_bt, c0_bt]) 
                  
        h = h0_bt.reshape(h0_bt.shape[0], -1, beam_num) # (b, *, beam_num)
        
        # Decoder
        h = self.Decoder(h)
        
        # Pooling
        h = nn.AvgPool1d(kernel_size = (h.shape[2]))(h) # (b, c_i, 1)
        h = torch.squeeze(h) # (b, c_i)
        
        # Output
        y = self.out_fc(h)
                
        # (b, BS_num * beam_num)
        
        return y
    
class Up(nn.Module):

    def __init__(self, his_len = 9, pre_len = 1, BS_num = 4, beam_num = 64, BS_dim = 'feature_map', device = 'cpu'):
        super(Up, self).__init__()
        
        self.his_len = his_len
        self.BS_num = BS_num
        self.beam_num = beam_num
        self.BS_dim = BS_dim
        self.device = device
        
        # Encoder
        self.Encoder = []       
        if BS_dim == 'feature_channel':
            print('you choose feature channel for BS dim!') 
            self.bn0 = nn.BatchNorm1d(2 * BS_num)
            self.Encoder.append(CB(c_i = 2 * BS_num, c_o = 64, k = 3, s = 3, p = 1))
        else:   
            print('you choose feature map for BS dim!')       
            self.bn0 = nn.BatchNorm1d(2)
            self.Encoder.append(CB(c_i = 2, c_o = 64, k = 3, s = 3, p = 1))
        self.Encoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1))
        self.Encoder.append(CB(c_i = 64, c_o = 512, k = 3, s = 3, p = 1))   
        # self.Encoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1)) 
        self.Encoder = nn.Sequential(*self.Encoder)
       
        # LSTM layer for extracting temporal correlations
        self.en_lstm = nn.LSTMCell(input_size = 512, hidden_size = 512)
                
        # Decoder
        self.Decoder = []        
        self.Decoder.append(CB(c_i = 16, c_o = 64, k = 3, s = 3, p = 1))
        self.Decoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1))
        self.Decoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1))  
        # self.Decoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1))              
        self.Decoder = nn.Sequential(*self.Decoder)
        
        # FC for output
        self.out_fc = nn.Linear(64, 2)

        
    def forward(self, x):
        
        # x: size (b, 2, his_len, BS_num, beam_num)
        _, _, his_len, BS_num, beam_num = x.shape
        
        if self.BS_dim == 'feature_channel':
            x = x.permute(0, 2, 1, 3, 4) # (b, his_len, 2, BS_num, beam_num)
            x = x.reshape(-1, 2, BS_num, beam_num) # (b * his_len, 2, BS_num, beam_num)
            x = x.reshape(x.shape[0], -1, beam_num) # (b * his_len, 2 * BS_num, beam_num)
        else:
            x = x.permute(0, 2, 1, 3, 4) # (b, his_len, 2, BS_num, beam_num)
            x = x.reshape(-1, 2, BS_num, beam_num) # (b * his_len, 2, BS_num, beam_num)
            x = x.reshape(x.shape[0], 2, -1) # (b * his_len, 2, BS_num * beam_num), note that beams for the same BS should be adjacent
                    
        # BN
        x = self.bn0(x)
        
        # Encoder
        x = self.Encoder(x)
        
        # Pooling
        x = nn.AvgPool1d(kernel_size = (x.shape[2]))(x) # (b * his_len, c_o, 1)
        x = torch.squeeze(x) # (b * his_len, c_o)
        x = x.reshape(-1, self.his_len, x.shape[1]) # (b, his_len, c_o)     
        x = x.permute(1, 0, 2) # (his_len, b, c_o)
        
        h0_Up = torch.zeros((x.shape[1], 512), device = self.device, requires_grad = True)
        c0_Up = torch.zeros((x.shape[1], 512), device = self.device, requires_grad = True)
        
        # LSTM learns temporal correlations
        for t in range(his_len):
          inputs_Up = x[t, :, :] # (b, c_o)
          h0_Up, c0_Up = self.en_lstm(inputs_Up, [h0_Up, c0_Up]) 
                  
        h = h0_Up.reshape(h0_Up.shape[0], -1, beam_num) # (b, *, beam_num)
        
        # Decoder
        h = self.Decoder(h)
        
        # Pooling
        h = nn.AvgPool1d(kernel_size = (h.shape[2]))(h) # (b, c_i, 1)
        h = torch.squeeze(h) # (b, c_i)
        
        # Output
        y = self.out_fc(h)
                
        # (b, 2)
        
        return y


class Vanilla(nn.Module):

    def __init__(self, his_len = 9, pre_len = 1, BS_num = 4, beam_num = 64, cnn_feature_num = 64, lstm_feature_num = 512,
                 BS_dim = 'feature_map', device = 'cpu'):
        super(Vanilla, self).__init__()
        
        self.his_len = his_len
        self.BS_num = BS_num
        self.beam_num = beam_num
        self.BS_dim = BS_dim
        self.device = device
        self.lstm_feature_num = lstm_feature_num
        
        # Encoder
        self.Encoder = []       
        if BS_dim == 'feature_channel':
            print('you choose feature channel for BS dim!') 
            self.bn0 = nn.BatchNorm1d(2 * BS_num)
            self.Encoder.append(CB(c_i = 2 * BS_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        else:   
            print('you choose feature map for BS dim!')       
            self.bn0 = nn.BatchNorm1d(2)
            self.Encoder.append(CB(c_i = 2, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Encoder.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Encoder.append(CB(c_i = cnn_feature_num, c_o = lstm_feature_num, k = 3, s = 3, p = 1))   
        # self.Encoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1)) 
        self.Encoder = nn.Sequential(*self.Encoder)
       
        # LSTM layer for extracting temporal correlations
        # self.ln_bt = nn.LayerNorm(128)
        self.caslstm_Bs = nn.LSTMCell(input_size = lstm_feature_num, hidden_size = lstm_feature_num)  
        self.caslstm_bt = nn.LSTMCell(input_size = lstm_feature_num, hidden_size = lstm_feature_num)
        self.caslstm_Up = nn.LSTMCell(input_size = lstm_feature_num, hidden_size = lstm_feature_num)
        # self.ln_up = nn.LayerNorm(beam_num * 2)      
        
        # Decoder for BS-selection
        self.Decoder_Bs = []        
        self.Decoder_Bs.append(CB(c_i = int(lstm_feature_num / beam_num), c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Bs.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Bs.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))              
        self.Decoder_Bs = nn.Sequential(*self.Decoder_Bs)
        
        # Output FC for BS-selection
        self.out_fc_Bs = nn.Linear(cnn_feature_num, BS_num)
                
        # Decoder for beam-tracking
        self.Decoder_bt = []        
        self.Decoder_bt.append(CB(c_i = int(lstm_feature_num / beam_num), c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_bt.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_bt.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))              
        self.Decoder_bt = nn.Sequential(*self.Decoder_bt)
        
        # Output FC for beam-tracking
        self.out_fc_bt = nn.Linear(cnn_feature_num, BS_num * beam_num)
        
        # Decoder for UE position
        self.Decoder_Up = []        
        self.Decoder_Up.append(CB(c_i = int(lstm_feature_num / beam_num), c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Up.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Up.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))              
        self.Decoder_Up = nn.Sequential(*self.Decoder_Up)
        
        # Multi-task cascaded
        # self.attention = CA(channel_i = beam_num * 2 + 64, channel_o = beam_num * 2)
        
        # Output FC for UE positioning
        self.out_fc_Up = nn.Linear(cnn_feature_num, 2)

        
    def forward(self, x):
        
        # x: size (b, 2, his_len, BS_num, beam_num)
        _, _, his_len, BS_num, beam_num = x.shape
        
        if self.BS_dim == 'feature_channel':
            x = x.permute(0, 2, 1, 3, 4) # (b, his_len, 2, BS_num, beam_num)
            x = x.reshape(-1, 2, BS_num, beam_num) # (b * his_len, 2, BS_num, beam_num)
            x = x.reshape(x.shape[0], -1, beam_num) # (b * his_len, 2 * BS_num, beam_num)
        else:
            x = x.permute(0, 2, 1, 3, 4) # (b, his_len, 2, BS_num, beam_num)
            x = x.reshape(-1, 2, BS_num, beam_num) # (b * his_len, 2, BS_num, beam_num)
            x = x.reshape(x.shape[0], 2, -1) # (b * his_len, 2, BS_num * beam_num), note that beams for the same BS should be adjacent
                    
        # BN
        x = self.bn0(x)
        
        # Encoder
        x = self.Encoder(x)
        
        # Pooling
        x = nn.AvgPool1d(kernel_size = (x.shape[2]))(x) # (b * his_len, c_o, 1)
        x = torch.squeeze(x) # (b * his_len, c_o)
        x = x.reshape(-1, self.his_len, x.shape[1]) # (b, his_len, c_o)     
        x = x.permute(1, 0, 2) # (his_len, b, c_o)
        
        h0_Bs = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_Bs = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        h0_bt = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_bt = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        h0_Up = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_Up = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        
        # LSTM learns temporal correlations
        for t in range(his_len):
          inputs_Bs = x[t, :, :] # (b, c_o)
          h0_Bs, c0_Bs = self.caslstm_Bs(inputs_Bs, [h0_Bs, c0_Bs]) 
          
          inputs_bt = x[t, :, :] # (b, c_o)
          h0_bt, c0_bt = self.caslstm_bt(inputs_bt, [h0_bt, c0_bt]) 
          
          inputs_Up = x[t, :, :] # (b, c_o)
          h0_Up, c0_Up = self.caslstm_Up(inputs_Up, [h0_Up, c0_Up]) 
        
        #######################################################
        # Decoder for BS-selection
        h_Bs = h0_Bs.reshape(h0_Bs.shape[0], -1, beam_num) # (b, *, beam_num)
        h_Bs = self.Decoder_Bs(h_Bs)
        
        # Pooling
        h_Bs = nn.AvgPool1d(kernel_size = (h_Bs.shape[2]))(h_Bs) # (b, c_i, 1)
        h_Bs = torch.squeeze(h_Bs) # (b, c_i)
        
        # Output
        y_Bs = self.out_fc_Bs(h_Bs)
                
        # (b, BS_num)
        #######################################################
        
        #######################################################
        # Decoder for beam-tracking
        h_bt = h0_bt.reshape(h0_bt.shape[0], -1, beam_num)
        h_bt = self.Decoder_bt(h_bt)
        
        # Pooling
        h_bt = nn.AvgPool1d(kernel_size = (h_bt.shape[2]))(h_bt) # (b, c_i, 1)
        h_bt = torch.squeeze(h_bt) # (b, c_i)
        
        # Output
        y_bt = self.out_fc_bt(h_bt)
                
        # (b, BS_num * beam_num)
        #######################################################
        
        #######################################################
        # Decoder for UE position
        h_Up = h0_Up.reshape(h0_Up.shape[0], -1, beam_num)        
        h_Up = self.Decoder_Up(h_Up)
        
        # Pooling
        h_Up = nn.AvgPool1d(kernel_size = (h_Up.shape[2]))(h_Up) # (b, c_i, 1)
        h_Up = torch.squeeze(h_Up) # (b, c_i)
        
        # Output
        y_Up = self.out_fc_Up(h_Up)
                
        # (b, 2)
        #######################################################
        
        return y_Bs, y_bt, y_Up

class Bs2bt2Up(nn.Module):

    def __init__(self, his_len = 9, pre_len = 1, BS_num = 4, beam_num = 64, cnn_feature_num = 64, lstm_feature_num = 512, cascaded_lstm_dropout = 0.3,
                 BS_dim = 'feature_map', device = 'cpu'):
        super(Bs2bt2Up, self).__init__()
        
        self.his_len = his_len
        self.BS_num = BS_num
        self.beam_num = beam_num
        self.BS_dim = BS_dim
        self.device = device
        self.lstm_feature_num = lstm_feature_num
        
        # Encoder
        self.Encoder = []       
        if BS_dim == 'feature_channel':
            print('you choose feature channel for BS dim!') 
            self.bn0 = nn.BatchNorm1d(2 * BS_num)
            self.Encoder.append(CB(c_i = 2 * BS_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        else:   
            print('you choose feature map for BS dim!')       
            self.bn0 = nn.BatchNorm1d(2)
            self.Encoder.append(CB(c_i = 2, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Encoder.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Encoder.append(CB(c_i = cnn_feature_num, c_o = lstm_feature_num, k = 3, s = 3, p = 1))   
        # self.Encoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1)) 
        self.Encoder = nn.Sequential(*self.Encoder)
       
        # LSTM layer for extracting temporal correlations
        # self.ln_bt = nn.LayerNorm(128)
        self.caslstm_Bs = nn.LSTMCell(input_size = lstm_feature_num, hidden_size = lstm_feature_num)  
        self.caslstm_bt = nn.LSTMCell(input_size = lstm_feature_num, hidden_size = lstm_feature_num)
        self.caslstm_Up = nn.LSTMCell(input_size = lstm_feature_num, hidden_size = lstm_feature_num)
        # self.ln_up = nn.LayerNorm(beam_num * 2)      
        self.dp_Bs2bt = nn.Dropout(cascaded_lstm_dropout)
        self.dp_bt2Up = nn.Dropout(cascaded_lstm_dropout)
        
        # Decoder for BS-selection
        self.Decoder_Bs = []        
        self.Decoder_Bs.append(CB(c_i = int(lstm_feature_num / beam_num), c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Bs.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Bs.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))              
        self.Decoder_Bs = nn.Sequential(*self.Decoder_Bs)
        
        # Output FC for BS-selection
        self.out_fc_Bs = nn.Linear(cnn_feature_num, BS_num)
                
        # Decoder for beam-tracking
        self.Decoder_bt = []        
        self.Decoder_bt.append(CB(c_i = int(lstm_feature_num / beam_num), c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_bt.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_bt.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))              
        self.Decoder_bt = nn.Sequential(*self.Decoder_bt)
        
        # Output FC for beam-tracking
        self.out_fc_bt = nn.Linear(cnn_feature_num, BS_num * beam_num)
        
        # Decoder for UE position
        self.Decoder_Up = []        
        self.Decoder_Up.append(CB(c_i = int(lstm_feature_num / beam_num), c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Up.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Up.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))              
        self.Decoder_Up = nn.Sequential(*self.Decoder_Up)
        
        # Multi-task cascaded
        # self.attention = CA(channel_i = beam_num * 2 + 64, channel_o = beam_num * 2)
        
        # Output FC for UE positioning
        self.out_fc_Up = nn.Linear(cnn_feature_num, 2)

        
    def forward(self, x):
        
        # x: size (b, 2, his_len, BS_num, beam_num)
        _, _, his_len, BS_num, beam_num = x.shape
        
        if self.BS_dim == 'feature_channel':
            x = x.permute(0, 2, 1, 3, 4) # (b, his_len, 2, BS_num, beam_num)
            x = x.reshape(-1, 2, BS_num, beam_num) # (b * his_len, 2, BS_num, beam_num)
            x = x.reshape(x.shape[0], -1, beam_num) # (b * his_len, 2 * BS_num, beam_num)
        else:
            x = x.permute(0, 2, 1, 3, 4) # (b, his_len, 2, BS_num, beam_num)
            x = x.reshape(-1, 2, BS_num, beam_num) # (b * his_len, 2, BS_num, beam_num)
            x = x.reshape(x.shape[0], 2, -1) # (b * his_len, 2, BS_num * beam_num), note that beams for the same BS should be adjacent
                    
        # BN
        x = self.bn0(x)
        
        # Encoder
        x = self.Encoder(x)
        
        # Pooling
        x = nn.AvgPool1d(kernel_size = (x.shape[2]))(x) # (b * his_len, c_o, 1)
        x = torch.squeeze(x) # (b * his_len, c_o)
        x = x.reshape(-1, self.his_len, x.shape[1]) # (b, his_len, c_o)     
        x = x.permute(1, 0, 2) # (his_len, b, c_o)
        
        h0_Bs = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_Bs = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        h0_bt = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_bt = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        h0_Up = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_Up = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        
        # LSTM learns temporal correlations
        for t in range(his_len):
          inputs_Bs = x[t, :, :] # (b, c_o)
          h0_Bs, c0_Bs = self.caslstm_Bs(inputs_Bs, [h0_Bs, c0_Bs]) 
          
          inputs_bt = inputs_Bs + self.dp_Bs2bt(h0_Bs) # (b, c_o)
          h0_bt, c0_bt = self.caslstm_bt(inputs_bt, [h0_bt, c0_bt]) 
          
          inputs_Up = inputs_bt + self.dp_bt2Up(h0_bt)
          h0_Up, c0_Up = self.caslstm_Up(inputs_Up, [h0_Up, c0_Up]) 
        
        #######################################################
        # Decoder for BS-selection
        h_Bs = h0_Bs.reshape(h0_Bs.shape[0], -1, beam_num) # (b, *, beam_num)
        h_Bs = self.Decoder_Bs(h_Bs)
        
        # Pooling
        h_Bs = nn.AvgPool1d(kernel_size = (h_Bs.shape[2]))(h_Bs) # (b, c_i, 1)
        h_Bs = torch.squeeze(h_Bs) # (b, c_i)
        
        # Output
        y_Bs = self.out_fc_Bs(h_Bs)
                
        # (b, BS_num)
        #######################################################
        
        #######################################################
        # Decoder for beam-tracking
        h_bt = h0_bt.reshape(h0_bt.shape[0], -1, beam_num)
        h_bt = self.Decoder_bt(h_bt)
        
        # Pooling
        h_bt = nn.AvgPool1d(kernel_size = (h_bt.shape[2]))(h_bt) # (b, c_i, 1)
        h_bt = torch.squeeze(h_bt) # (b, c_i)
        
        # Output
        y_bt = self.out_fc_bt(h_bt)
                
        # (b, BS_num * beam_num)
        #######################################################
        
        #######################################################
        # Decoder for UE position
        h_Up = h0_Up.reshape(h0_Up.shape[0], -1, beam_num)        
        h_Up = self.Decoder_Up(h_Up)
        
        # Pooling
        h_Up = nn.AvgPool1d(kernel_size = (h_Up.shape[2]))(h_Up) # (b, c_i, 1)
        h_Up = torch.squeeze(h_Up) # (b, c_i)
        
        # Output
        y_Up = self.out_fc_Up(h_Up)
                
        # (b, 2)
        #######################################################
        
        return y_Bs, y_bt, y_Up
    
class Up2bt2Bs(nn.Module):

    def __init__(self, his_len = 9, pre_len = 1, BS_num = 4, beam_num = 64, cnn_feature_num = 64, lstm_feature_num = 512, cascaded_lstm_dropout = 0.3,
                 BS_dim = 'feature_map', device = 'cpu'):
        super(Up2bt2Bs, self).__init__()
        
        self.his_len = his_len
        self.BS_num = BS_num
        self.beam_num = beam_num
        self.BS_dim = BS_dim
        self.device = device
        self.lstm_feature_num = lstm_feature_num
        
        # Encoder
        self.Encoder = []       
        if BS_dim == 'feature_channel':
            print('you choose feature channel for BS dim!') 
            self.bn0 = nn.BatchNorm1d(2 * BS_num)
            self.Encoder.append(CB(c_i = 2 * BS_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        else:   
            print('you choose feature map for BS dim!')       
            self.bn0 = nn.BatchNorm1d(2)
            self.Encoder.append(CB(c_i = 2, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Encoder.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Encoder.append(CB(c_i = cnn_feature_num, c_o = lstm_feature_num, k = 3, s = 3, p = 1))   
        # self.Encoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1)) 
        self.Encoder = nn.Sequential(*self.Encoder)
       
        # LSTM layer for extracting temporal correlations
        # self.ln_bt = nn.LayerNorm(128)
        self.caslstm_Bs = nn.LSTMCell(input_size = lstm_feature_num, hidden_size = lstm_feature_num)  
        self.caslstm_bt = nn.LSTMCell(input_size = lstm_feature_num, hidden_size = lstm_feature_num)
        self.caslstm_Up = nn.LSTMCell(input_size = lstm_feature_num, hidden_size = lstm_feature_num)
        # self.ln_up = nn.LayerNorm(beam_num * 2)      
        self.dp_Up2bt = nn.Dropout(cascaded_lstm_dropout)
        self.dp_bt2Bs = nn.Dropout(cascaded_lstm_dropout)
        
        # Decoder for BS-selection
        self.Decoder_Bs = []        
        self.Decoder_Bs.append(CB(c_i = int(lstm_feature_num / beam_num), c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Bs.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Bs.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))              
        self.Decoder_Bs = nn.Sequential(*self.Decoder_Bs)
        
        # Output FC for BS-selection
        self.out_fc_Bs = nn.Linear(cnn_feature_num, BS_num)
                
        # Decoder for beam-tracking
        self.Decoder_bt = []        
        self.Decoder_bt.append(CB(c_i = int(lstm_feature_num / beam_num), c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_bt.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_bt.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))              
        self.Decoder_bt = nn.Sequential(*self.Decoder_bt)
        
        # Output FC for beam-tracking
        self.out_fc_bt = nn.Linear(cnn_feature_num, BS_num * beam_num)
        
        # Decoder for UE position
        self.Decoder_Up = []        
        self.Decoder_Up.append(CB(c_i = int(lstm_feature_num / beam_num), c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Up.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Up.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))              
        self.Decoder_Up = nn.Sequential(*self.Decoder_Up)
        
        # Multi-task cascaded
        # self.attention = CA(channel_i = beam_num * 2 + 64, channel_o = beam_num * 2)
        
        # Output FC for UE positioning
        self.out_fc_Up = nn.Linear(cnn_feature_num, 2)

        
    def forward(self, x):
        
        # x: size (b, 2, his_len, BS_num, beam_num)
        _, _, his_len, BS_num, beam_num = x.shape
        
        if self.BS_dim == 'feature_channel':
            x = x.permute(0, 2, 1, 3, 4) # (b, his_len, 2, BS_num, beam_num)
            x = x.reshape(-1, 2, BS_num, beam_num) # (b * his_len, 2, BS_num, beam_num)
            x = x.reshape(x.shape[0], -1, beam_num) # (b * his_len, 2 * BS_num, beam_num)
        else:
            x = x.permute(0, 2, 1, 3, 4) # (b, his_len, 2, BS_num, beam_num)
            x = x.reshape(-1, 2, BS_num, beam_num) # (b * his_len, 2, BS_num, beam_num)
            x = x.reshape(x.shape[0], 2, -1) # (b * his_len, 2, BS_num * beam_num), note that beams for the same BS should be adjacent
                    
        # BN
        x = self.bn0(x)
        
        # Encoder
        x = self.Encoder(x)
        
        # Pooling
        x = nn.AvgPool1d(kernel_size = (x.shape[2]))(x) # (b * his_len, c_o, 1)
        x = torch.squeeze(x) # (b * his_len, c_o)
        x = x.reshape(-1, self.his_len, x.shape[1]) # (b, his_len, c_o)     
        x = x.permute(1, 0, 2) # (his_len, b, c_o)
        
        h0_Bs = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_Bs = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        h0_bt = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_bt = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        h0_Up = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_Up = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        
        # LSTM learns temporal correlations
        for t in range(his_len):
          inputs_Up = x[t, :, :] # (b, c_o)
          h0_Up, c0_Up = self.caslstm_Up(inputs_Up, [h0_Up, c0_Up]) 
          
          inputs_bt = inputs_Up + self.dp_Up2bt(h0_Up) # (b, c_o)
          h0_bt, c0_bt = self.caslstm_bt(inputs_bt, [h0_bt, c0_bt]) 
          
          inputs_Bs = inputs_bt + self.dp_bt2Bs(h0_bt)
          h0_Bs, c0_Bs = self.caslstm_Bs(inputs_Bs, [h0_Bs, c0_Bs]) 
        
        #######################################################
        # Decoder for BS-selection
        h_Bs = h0_Bs.reshape(h0_Bs.shape[0], -1, beam_num) # (b, *, beam_num)
        h_Bs = self.Decoder_Bs(h_Bs)
        
        # Pooling
        h_Bs = nn.AvgPool1d(kernel_size = (h_Bs.shape[2]))(h_Bs) # (b, c_i, 1)
        h_Bs = torch.squeeze(h_Bs) # (b, c_i)
        
        # Output
        y_Bs = self.out_fc_Bs(h_Bs)
                
        # (b, BS_num)
        #######################################################
        
        #######################################################
        # Decoder for beam-tracking
        h_bt = h0_bt.reshape(h0_bt.shape[0], -1, beam_num)
        h_bt = self.Decoder_bt(h_bt)
        
        # Pooling
        h_bt = nn.AvgPool1d(kernel_size = (h_bt.shape[2]))(h_bt) # (b, c_i, 1)
        h_bt = torch.squeeze(h_bt) # (b, c_i)
        
        # Output
        y_bt = self.out_fc_bt(h_bt)
                
        # (b, BS_num * beam_num)
        #######################################################
        
        #######################################################
        # Decoder for UE position
        h_Up = h0_Up.reshape(h0_Up.shape[0], -1, beam_num)        
        h_Up = self.Decoder_Up(h_Up)
        
        # Pooling
        h_Up = nn.AvgPool1d(kernel_size = (h_Up.shape[2]))(h_Up) # (b, c_i, 1)
        h_Up = torch.squeeze(h_Up) # (b, c_i)
        
        # Output
        y_Up = self.out_fc_Up(h_Up)
                
        # (b, 2)
        #######################################################
        
        return y_Bs, y_bt, y_Up

class Dual_Cascaded(nn.Module):

    def __init__(self, his_len = 9, pre_len = 1, BS_num = 4, beam_num = 64, cnn_feature_num = 64, lstm_feature_num = 512, cascaded_lstm_dropout = 0.3,
                 BS_dim = 'feature_map', device = 'cpu'):
        super(Dual_Cascaded, self).__init__()
        
        self.his_len = his_len
        self.BS_num = BS_num
        self.beam_num = beam_num
        self.BS_dim = BS_dim
        self.device = device
        
        self.lstm_feature_num = lstm_feature_num
        
        # Encoder
        self.Encoder = []       
        if BS_dim == 'feature_channel':
            print('you choose feature channel for BS dim!') 
            self.bn0 = nn.BatchNorm1d(2 * BS_num)
            self.Encoder.append(CB(c_i = 2 * BS_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        else:   
            print('you choose feature map for BS dim!')       
            self.bn0 = nn.BatchNorm1d(2)
            self.Encoder.append(CB(c_i = 2, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Encoder.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Encoder.append(CB(c_i = cnn_feature_num, c_o = lstm_feature_num, k = 3, s = 3, p = 1))   
        # self.Encoder.append(CB(c_i = 64, c_o = 64, k = 3, s = 3, p = 1)) 
        self.Encoder = nn.Sequential(*self.Encoder)
       
        # LSTM layer for extracting temporal correlations
        self.caslstm_Bs = nn.LSTMCell(input_size = lstm_feature_num, hidden_size = lstm_feature_num)  
        self.caslstm_bt = nn.LSTMCell(input_size = lstm_feature_num, hidden_size = lstm_feature_num)
        self.caslstm_Up = nn.LSTMCell(input_size = lstm_feature_num, hidden_size = lstm_feature_num)     
        self.dp_Bs2bt = nn.Dropout(cascaded_lstm_dropout)
        self.dp_bt2Up = nn.Dropout(cascaded_lstm_dropout)
        self.dp_Up2Bs = nn.Dropout(cascaded_lstm_dropout)
                
        # self.weight_Bs = nn.Parameter(torch.tensor(0.5, dtype = torch.float))
        # self.weight_bt = nn.Parameter(torch.tensor(0.5, dtype = torch.float))
        # self.weight_Up = nn.Parameter(torch.tensor(0.5, dtype = torch.float))
        
        # Decoder for BS-selection
        self.Decoder_Bs = []        
        self.Decoder_Bs.append(CB(c_i = int(lstm_feature_num / beam_num), c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Bs.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Bs.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))              
        self.Decoder_Bs = nn.Sequential(*self.Decoder_Bs)
        
        # Output FC for BS-selection
        self.out_fc_Bs = nn.Linear(cnn_feature_num, BS_num)
                
        # Decoder for beam-tracking
        self.Decoder_bt = []        
        self.Decoder_bt.append(CB(c_i = int(lstm_feature_num / beam_num), c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_bt.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_bt.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))              
        self.Decoder_bt = nn.Sequential(*self.Decoder_bt)
        
        # Output FC for beam-tracking
        self.out_fc_bt = nn.Linear(cnn_feature_num, BS_num * beam_num)
        
        # Decoder for UE position
        self.Decoder_Up = []        
        self.Decoder_Up.append(CB(c_i = int(lstm_feature_num / beam_num), c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Up.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))
        self.Decoder_Up.append(CB(c_i = cnn_feature_num, c_o = cnn_feature_num, k = 3, s = 3, p = 1))              
        self.Decoder_Up = nn.Sequential(*self.Decoder_Up)
        
        # Multi-task cascaded
        # self.attention = CA(channel_i = beam_num * 2 + 64, channel_o = beam_num * 2)
        
        # Output FC for UE positioning
        self.out_fc_Up = nn.Linear(cnn_feature_num, 2)

        
    def forward(self, x):
        
        # x: size (b, 2, his_len, BS_num, beam_num)
        _, _, his_len, BS_num, beam_num = x.shape
        
        if self.BS_dim == 'feature_channel':
            x = x.permute(0, 2, 1, 3, 4) # (b, his_len, 2, BS_num, beam_num)
            x = x.reshape(-1, 2, BS_num, beam_num) # (b * his_len, 2, BS_num, beam_num)
            x = x.reshape(x.shape[0], -1, beam_num) # (b * his_len, 2 * BS_num, beam_num)
        else:
            x = x.permute(0, 2, 1, 3, 4) # (b, his_len, 2, BS_num, beam_num)
            x = x.reshape(-1, 2, BS_num, beam_num) # (b * his_len, 2, BS_num, beam_num)
            x = x.reshape(x.shape[0], 2, -1) # (b * his_len, 2, BS_num * beam_num), note that beams for the same BS should be adjacent
                    
        # BN
        x = self.bn0(x)
        
        # Encoder
        x = self.Encoder(x)
        
        # Pooling
        x = nn.AvgPool1d(kernel_size = (x.shape[2]))(x) # (b * his_len, c_o, 1)
        x = torch.squeeze(x) # (b * his_len, c_o)
        x = x.reshape(-1, self.his_len, x.shape[1]) # (b, his_len, c_o)     
        x = x.permute(1, 0, 2) # (his_len, b, c_o)
        
        h0_Bs_1 = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_Bs_1 = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        h0_bt_1 = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_bt_1 = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        h0_Up_1 = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_Up_1 = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        
        h0_Bs_2 = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_Bs_2 = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True) 
        h0_bt_2 = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_bt_2 = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        h0_Up_2 = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
        c0_Up_2 = torch.zeros((x.shape[1], self.lstm_feature_num), device = self.device, requires_grad = True)
                
        # LSTM learns temporal correlations
        for t in range(his_len):
            #######################################################
            # direction: shared feature -> Bs -> bt -> Up
            inputs_Bs = x[t, :, :] # (b, c_o)
            h0_Bs_1, c0_Bs_1 = self.caslstm_Bs(inputs_Bs, [h0_Bs_1, c0_Bs_1]) 
            
            inputs_bt = inputs_Bs + self.dp_Bs2bt(h0_Bs_1) # (b, c_o)
            h0_bt_1, c0_bt_1 = self.caslstm_bt(inputs_bt, [h0_bt_1, c0_bt_1]) 
            
            inputs_Up = inputs_bt + self.dp_bt2Up(h0_bt_1) # (b, c_o)
            h0_Up_1, c0_Up_1 = self.caslstm_Up(inputs_Up, [h0_Up_1, c0_Up_1]) 
            #######################################################
            
            #######################################################
            # direction: Up -> Bs -> bt -> Up
            inputs_Bs = inputs_Up + self.dp_Up2Bs(h0_Up_1) # (b, c_o)
            h0_Bs_2, c0_Bs_2 = self.caslstm_Bs(inputs_Bs, [h0_Bs_2, c0_Bs_2]) 
            
            inputs_bt = inputs_Bs + self.dp_Bs2bt(h0_Bs_2) # (b, c_o)
            h0_bt_2, c0_bt_2 = self.caslstm_bt(inputs_bt, [h0_bt_2, c0_bt_2]) 
            
            inputs_Up = inputs_bt + self.dp_bt2Up(h0_bt_2) # (b, c_o)
            h0_Up_2, c0_Up_2 = self.caslstm_Up(inputs_Up, [h0_Up_2, c0_Up_2]) 
            #######################################################
                    
        h0_Bs = h0_Bs_2 # (b, c_o)
        h0_bt = h0_bt_2 # (b, c_o)
        h0_Up = h0_Up_2 # (b, c_o)
        
        #######################################################
        # Decoder for BS-selection
        h_Bs = h0_Bs.reshape(h0_Bs.shape[0], -1, beam_num) # (b, *, beam_num)
        h_Bs = self.Decoder_Bs(h_Bs)
        
        # Pooling
        h_Bs = nn.AvgPool1d(kernel_size = (h_Bs.shape[2]))(h_Bs) # (b, c_i, 1)
        h_Bs = torch.squeeze(h_Bs) # (b, c_i)
        
        # Output
        y_Bs = self.out_fc_Bs(h_Bs)
                
        # (b, BS_num)
        #######################################################
        
        #######################################################
        # Decoder for beam-tracking
        h_bt = h0_bt.reshape(h0_bt.shape[0], -1, beam_num)
        h_bt = self.Decoder_bt(h_bt)
        
        # Pooling
        h_bt = nn.AvgPool1d(kernel_size = (h_bt.shape[2]))(h_bt) # (b, c_i, 1)
        h_bt = torch.squeeze(h_bt) # (b, c_i)
        
        # Output
        y_bt = self.out_fc_bt(h_bt)
                
        # (b, BS_num * beam_num)
        #######################################################
        
        #######################################################
        # Decoder for UE position
        h_Up = h0_Up.reshape(h0_Up.shape[0], -1, beam_num)        
        h_Up = self.Decoder_Up(h_Up)
        
        # Pooling
        h_Up = nn.AvgPool1d(kernel_size = (h_Up.shape[2]))(h_Up) # (b, c_i, 1)
        h_Up = torch.squeeze(h_Up) # (b, c_i)
        
        # Output
        y_Up = self.out_fc_Up(h_Up)
                
        # (b, 2)
        #######################################################
        
        return y_Bs, y_bt, y_Up
    
    