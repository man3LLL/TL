import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch
import torch.nn as nn
import torch.fft

import numpy as np


class Transformer(nn.Module):
    def __init__(self, feature_size=7, num_layers=1, dropout=0.2):#dropout=0.2
        super(Transformer, self).__init__()

        # Define the encoder layer
        # This consists of a multi-head self-attention mechanism followed by a simple, position-wise fully connected feed-forward network.
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,  # The number of expected features in the input
            nhead=1,  # The number of heads in the multihead attention models
            dropout=dropout,  # The dropout value (default=0.5)
            dim_feedforward=2048  #2048
        )

        # Define the transformer encoder
        # This is a stack of N encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers  # The number of sub-encoder-layers in the encoder
        )

        # Define the decoder
        # A linear layer to map the output of the transformer encoder to a single value
        self.decoder = nn.Linear(
            feature_size,  # The number of expected features in the input
            7  # The number of features in the output
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize the weights and biases of the decoder to ensure that they are neither too small nor too large
        initrange = 0.1
        self.decoder.bias.data.zero_()  # Set biases to zero
        self.decoder.weight.data.uniform_(-initrange,
                                          initrange)  # Set weights to random values in the range [-0.1, 0.1]

    def _generate_square_subsequent_mask(self, sz):
        # Generate a mask to ensure that the self-attention mechanism only attends to positions that precede or are at the current position in the sequence
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, device):
        # The forward propagation method
        # src: the input tensor
        # device: the device on which to perform computations

        # Generate a mask and move it to the specified device
        mask = self._generate_square_subsequent_mask(len(src)).to(device)

        # Pass the input tensor through the transformer encoder
        # This will apply a series of self-attention and feed-forward operations to the input tensor
        output = self.transformer_encoder(src, mask)

        # Pass the output of the transformer encoder through the decoder to produce the final output
        output = self.decoder(output)
        return output


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):  # x 一般为(batch_size,input length/output length,chnnel]
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# class Model(nn.Module):
#     """
#     Decomposition-Linear
#     """
#
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#
#         # Decompsition Kernel Size
#         kernel_size = 25
#         self.decompsition = series_decomp(kernel_size)
#         self.individual = configs.individual
#         self.channels = configs.enc_in
#
#         if self.individual:
#             self.Linear_Seasonal = nn.ModuleList()
#             self.Linear_Trend = nn.ModuleList()
#
#             for i in range(self.channels):
#                 self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
#                 self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
#
#                 # Use this two lines if you want to visualize the weights
#                 # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#                 # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#         else:
#             self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
#             self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
#
#             # Use this two lines if you want to visualize the weights
#             # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#             # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#
#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#         seasonal_init, trend_init = self.decompsition(x)  # res=季节性  moving_mean=趋势性
#         seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2,
#                                                                                        1)  # (batch_size,channel,out length)
#         if self.individual:
#             seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
#                                           dtype=seasonal_init.dtype).to(seasonal_init.device)
#             trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
#                                        dtype=trend_init.dtype).to(trend_init.device)
#             for i in range(self.channels):
#                 seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
#                 trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
#         else:
#             seasonal_output = self.Linear_Seasonal(seasonal_init)
#             trend_output = self.Linear_Trend(trend_init)
#
#         x = seasonal_output + trend_output
#         return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]



class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()


        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = False
        self.channels = configs.enc_in
        self.transformer=Transformer()
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark):

        # z = self.rev(x, 'norm') # B, L, D -> B, L, D
        # print("xshape is",x.shape)
        # seasonal_init, trend_init = self.decompsition(x)  # res=季节性  moving_mean=趋势性

        # zeros_tensor = torch.zeros((32, 96, 7), dtype=seasonal_init.dtype, device='cuda')#除了96以外
        # seasonal_init_extended = torch.cat((seasonal_init, zeros_tensor), dim=1)


        # seasonal_init_extend=torch.cat([seasonal_init,seasonal_init],dim=1) #这句是将维度变成96+96=192

        # seasonal_init_extend = torch.cat([seasonal_init] * 3, dim=1)
        # zeros_tensor = torch.zeros((32, 48, 7), dtype=seasonal_init.dtype, device='cuda')
        # seasonal_init_extend = torch.cat((seasonal_init_extend, zeros_tensor), dim=1)#336

        # seasonal_init_extend = torch.cat([seasonal_init]*7 , dim=1)
        # zeros_tensor = torch.zeros((32, 48, 7), dtype=seasonal_init.dtype, device='cuda')
        # seasonal_init_extend = torch.cat((seasonal_init_extend, zeros_tensor), dim=1)#720

        #消融
        x = torch.cat([x] * 7, dim=1)
        zeros_tensor = torch.zeros((32, 48, 7), dtype=x.dtype, device='cuda')
        x= torch.cat((x, zeros_tensor), dim=1)#
        transout = self.transformer(x, 'cuda')

        # transout = self.transformer(seasonal_init, 'cuda') #不是96就改为extend
        transout = transout.permute(0,2,1)
        # seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2,1)  # (batch_size,channel,out length)
        # if self.individual:
        #     seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
        #                                   dtype=seasonal_init.dtype).to(seasonal_init.device)
        #     trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
        #                                dtype=trend_init.dtype).to(trend_init.device)
        #     for i in range(self.channels):
        #         seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
        #         trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        # else:
        #     seasonal_output = self.Linear_Seasonal(seasonal_init)
        #     trend_output = self.Linear_Trend(trend_init)
            #print("not individual")
        # print("trans shape",transout.shape)
        # seasonal_output1=seasonal_output.permute(0,2,1)

        # transout =.permute(0, 2, 1)
        # print("sesonal_out",seasonal_output.shape)
        #z = seasonal_output + trend_output +transout

        z = transout
        # print("z is",z.shape)
        z = z.permute(0, 2, 1)
        # z = self.transformer(z, 'cuda')
        # print("trans ", z.shape)
        # z = z.permute(0, 2, 1)
        # z = self.backbone(z) # B, L, D -> B, H, D
        # z = z.permute(0, 2, 1)

        # z = self.rev(z, 'denorm') # B, H, D -> B, H, D

        # print("z shape is",z.shape)   #B,H,D=256,96,7
        return z




# class Model(nn.Module):
#     """
#     Vanilla Transformer with O(L^2) complexity
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention
#
#         # Embedding
#         self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                            configs.dropout)
#         self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                            configs.dropout)
#         # Encoder
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=configs.output_attention), configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )
#         # Decoder
#         self.decoder = Decoder(
#             [
#                 DecoderLayer(
#                     AttentionLayer(
#                         FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
#                         configs.d_model, configs.n_heads),
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
#                         configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation,
#                 )
#                 for l in range(configs.d_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model),
#             projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
#         )
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
#                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
#
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
#
#         dec_out = self.dec_embedding(x_dec, x_mark_dec)
#         dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
#
#         if self.output_attention:
#             return dec_out[:, -self.pred_len:, :], attns
#         else:
#             return dec_out[:, -self.pred_len:, :]
