import torch
import torch.nn as nn


class Conv(torch.nn.Module):
    """
    A convolution with the option to be causal and use xavier initialization
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        is_causal=False,
        device="cpu",
    ):
        super(Conv, self).__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        ).to(device)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        if self.is_causal:
            padding = (int((self.kernel_size - 1) * (self.dilation)), 0)
            signal = nn.functional.pad(signal, padding)
        return self.conv(signal)


class DilationConvLayer(nn.Module):
    #            |----------------------------------------|     *residual*
    #            |                                        |
    #            |    |-- conv -- tanh --|                |
    # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
    #                 |-- conv -- sigm --|     |
    #                                         1x1
    #                                          |
    # ---------------------------------------> + ------------->	*skip*

    def __init__(
        self, n_residual_channels, n_skip_channels, nr_layers, stack_time, device
    ):
        super().__init__()

        self.nr_layers = nr_layers
        self.dilation_rates = [2 ** i for i in range(nr_layers)] * stack_time
        self.n_residual_channels = n_residual_channels
        self.n_skip_channels = n_skip_channels

        dilation_conv_l = []
        skip_layer_l = []
        res_layer_l = []
        for dilation in self.dilation_rates:
            dilation_conv_l.append(
                Conv(
                    n_residual_channels,
                    2 * n_residual_channels,
                    kernel_size=2,
                    dilation=dilation,
                    w_init_gain="tanh",
                    is_causal=True,
                    device=device,
                )
            )
            # skip layer has kernal size 1, output dim is same as input
            skip_layer_l.append(
                Conv(
                    n_residual_channels,
                    n_skip_channels,
                    w_init_gain="relu",
                    device=device,
                )
            )

            # ------------------
            # for res_layer, last one is not necessary, just save some time
            # if i < nr_layers - 1:
            # ----------------
            # residual layer has kernal size 1, output dim is same as input
            res_layer_l.append(
                Conv(
                    n_residual_channels,
                    n_residual_channels,
                    w_init_gain="linear",
                    device=device,
                )
            )

        self.dilation_conv_l = nn.ModuleList(dilation_conv_l)
        self.skip_layer_l = nn.ModuleList(skip_layer_l)
        self.res_layer_l = nn.ModuleList(res_layer_l)

    def forward(self, forward_input):
        for i in range(len(self.dilation_rates)):
            x = self.dilation_conv_l[i](forward_input)

            # first half goes into filter convolution
            x_f = torch.tanh(x[:, : self.n_residual_channels, :])

            # second half goes into gating convolution
            x_g = torch.sigmoid(x[:, self.n_residual_channels:, :])

            # multiply filter and gating branches
            z = x_f * x_g

            # ------
            # if i < len(self.res_layer_l):
            # -----
            # print('size before reslayer', z.size())
            residual = self.res_layer_l[i](z)
            # print('size after reslayer', residual.size())

            # N.B. what about the last one?
            forward_input = forward_input + residual

            if i == 0:
                output = self.skip_layer_l[i](z)
            else:
                output = self.skip_layer_l[i](z) + output

        return output


class WaveNet(nn.Module):
    def __init__(
        self,
        n_in_channels,
        n_residual_channels,
        n_skip_channels,
        n_out_channels,
        nr_layers,
        stack_time,
        decode_len,
        device,
    ):
        super().__init__()

        self.decode_len = decode_len
        self.n_out_channels = n_out_channels
        self.device = device

        self.conv_start = Conv(
            n_in_channels,
            n_residual_channels,
            bias=False,
            w_init_gain="relu",
            device=device,
        )
        self.dilation_conv = DilationConvLayer(
            n_residual_channels, n_skip_channels, nr_layers, stack_time, device
        )
        self.conv_out = Conv(
            n_skip_channels,
            n_out_channels,
            bias=False,
            w_init_gain="relu",
            device=device,
        )
        self.conv_end = Conv(
            n_out_channels,
            n_out_channels,
            bias=False,
            w_init_gain="linear",
            device=device,
        )

    def forward(self, forward_input):
        """
        In training stage, we use force teaching
                  |------- encode ts------|
        .                                    |- decode ts -|
        input:    | | | | | | | | | | | | | 0 1 2 3 4 5 6 7
                  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX  --> Wavenet
                  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        output:                           0 1 2 3 4 5 6 7 8
        .                                  |-- decode ts --|

        forward_input size: [batch_size, input_dim, encode_len + decode_len -1]
        output size: [batch_size, input_dim, decode_len]
        """

        # output size: [batch_size, input_dim+3, encode_len+decode_len-1]
        x = self.conv_start(forward_input)
        output = self.dilation_conv(x)

        output = nn.functional.relu(output, True)
        # output size: [batch_size, 1, encode_len+decode_len-1]
        output = self.conv_out(output)

        output = nn.functional.relu(output, True)
        # output size: [batch_size, 1, encode_len+decode_len-1]
        output = self.conv_end(output)

        # only select the last decode_len as output
        # output dimension as [batch_size, decode_dim, decode_len]
        l = self.decode_len
        output = output[:, :, -l:]

        return output

    def predict_sequence(self, input_tensor):
        # In prediction stage, we predict ts one by one.
        # The newly predicted value will be appended to input ts
        #
        #           |------- encode ts -----|------ only features ----------|
        # .
        # input:    | | | | | | | | | | | | |   0   1   2   3   4   5   6   7
        #           XXXXXXXXXXXXXXXXXXXXXXXXX  /|  /|  /|  /|  /|  /|  /|  /|
        #           XXXXXXXXXXXXXXXXXXXXXXXXX / | / | / | / | / | / | / | / |
        #           XXXXXXXXXXXXXXXXXXXXXXXXX/  |/  |/  |/  |/  |/  |/  |/  |
        # output:                           0   1   2   3   4   5   6   7   8
        # .                                  |---------- predicted ts -------|
        #

        # input_tensor size: [batch_size, input_dim, encode_len + decode_len - 1]
        # N.B. for 1st dimension of decode_len part is all zero
        # output size: (batch_size, decode_dim, decode_len)

        decode_len = self.decode_len
        batch_size = len(input_tensor)
        decode_dim = self.n_out_channels

        # initialize output (pred_steps time steps)
        pred_sequence = torch.zeros(batch_size, decode_dim, decode_len).to(self.device)

        # inital input is only encode part
        history_tensor = input_tensor[:, :, : -(decode_len - 1)]

        for i in range(decode_len):
            # record next time step prediction (last time step of model output)
            last_step_pred = self.forward(history_tensor)[:, :, -1]

            pred_sequence[:, :, i] = last_step_pred

            # add the next time step prediction along with corresponding exogenous features to the history tensor
            last_step_exog = input_tensor[:, decode_dim:, [(-decode_len + 1) + i]]
            last_step_tensor = torch.cat(
                [last_step_pred.unsqueeze(2), last_step_exog], axis=1
            )
            history_tensor = torch.cat([history_tensor, last_step_tensor], axis=2)

        return pred_sequence


class WaveNetTS(nn.Module):
    def __init__(
        self, wavenet, cat_emb_layer, fixed_emb_layer, device
    ):
        super().__init__()
        self.wavenet = wavenet.to(device)
        self.cat_emb_layer = cat_emb_layer.to(device)
        self.fixed_emb_layer = fixed_emb_layer.to(device)
        self.device = device

    def get_embedding(
        self,
        src_ts,
        trg_ts,
        src_xdaysago,
        trg_xdaysago,
        cat_encode,
        cat_decode,
        fixed_feat
    ):
        # src_ts size: [encode_len, batch_size, 1]
        # trg_ts size: [decode_len, batch_size, 1]
        # src_xdaysago size: [encode_len, batch_size, history_data_dim]
        # trg_xdaysago size: [decode_len, batch_size, history_data_dim]
        # cat_encode size: [encode_len, batch_size, nr_cat_features]
        # cat_decode size: [decode_len, batch_size, nr_cat_features]
        # fixed_feat size: [batch_size, nr_fixed_features]

        encode_len = src_ts.shape[0]
        decode_len = trg_ts.shape[0]

        # encode_len = src_ts.shape[1]
        # decode_len = trg_ts.shape[1]
        # batch_size = trg_ts.shape[0]
        # trg_dim = trg_ts.shape[2]

        # categorical feature embedding
        # cat_en(de)code_emb: [en(de)code_len, batch_size, cat_feat_embedding_dim]
        cat_encode_emb = self.cat_emb_layer(cat_encode).to(self.device)
        cat_decode_emb = self.cat_emb_layer(cat_decode).to(self.device)

        # fixed feature embedding, fixed_emb size: [1, batch_size, all_emb_dim]
        fixed_emb = self.fixed_emb_layer(fixed_feat).unsqueeze(0).to(self.device)

        # encode_input size: [encode_len, batch_size, (ts_dim + xdaysago_dim + cat_emb_dim + fixed_emb_dim)]
        encode_input = torch.cat(
            [src_ts, src_xdaysago, cat_encode_emb, fixed_emb.repeat(encode_len, 1, 1)],
            2,
        )

        # decode_input size: [decode_len, batch_size, (ts_dim + xdaysago_dim + cat_emb_dim + fixed_emb_dim)]
        decode_input = torch.cat(
            [trg_ts, trg_xdaysago, cat_decode_emb, fixed_emb.repeat(decode_len, 1, 1)],
            2,
        )

        return encode_input, decode_input

    def merge_encode_decode_seq(self, encode_input, decode_input):
        # In wavenet, the required size is: [batch_size, input_dim, sequence_len]
        # so we permute the dimension
        encode_input = encode_input.permute(1, 2, 0)
        decode_input = decode_input.permute(1, 2, 0)

        # for Wavenet, input is both encode plus decode, but WITHOUT the last step of decode!!!
        decode_input = decode_input[:, :, :-1]

        # forward_input size: [batch_size, (ts_dim + xdaysago_dim + cat_emb_dim + fixed_emb_dim), encode_len+decode_len-1]
        forward_input = torch.cat([encode_input, decode_input], 2)

        return forward_input

    def forward(
        self,
        src_ts,
        trg_ts,
        src_xdaysago,
        trg_xdaysago,
        cat_encode,
        cat_decode,
        fixed_feat,
        teacher_forcing_ratio=None  # not like Seq2Seq, WaveNet does not need teaching force ratio
    ):
        # build embedding tensors and concatenate them
        encode_input, decode_input = self.get_embedding(
            src_ts,
            trg_ts,
            src_xdaysago,
            trg_xdaysago,
            cat_encode,
            cat_decode,
            fixed_feat,
        )

        # forward_input size: [batch_size, (ts_dim + xdaysago_dim + cat_emb_dim + fixed_emb_dim), encode_len+decode_len-1]
        forward_input = self.merge_encode_decode_seq(encode_input, decode_input)

        # output size: [batch_size, 1, decode_len]
        output = self.wavenet(forward_input)
        
        # change size to [decode_len, batch_size, 1] to match target tensor when compute loss
        output = output.permute(2, 0, 1)

        return output

    def generate(
        self,
        src_ts,
        trg_ts,
        src_xdaysago,
        trg_xdaysago,
        cat_encode,
        cat_decode,
        fixed_feat,
    ):
        """
        Make prediction
        """

        # forward_input size [batch_size, (ts_dim + xdaysago_dim + cat_emb_dim + fixed_emb_dim), encode_len+decode_len-1]
        encode_input, decode_input = self.get_embedding(
            src_ts,
            trg_ts,
            src_xdaysago,
            trg_xdaysago,
            cat_encode,
            cat_decode,
            fixed_feat,
        )

        # forward_input size: [batch_size, (ts_dim + xdaysago_dim + cat_emb_dim + fixed_emb_dim), encode_len+decode_len-1]
        forward_input = self.merge_encode_decode_seq(encode_input, decode_input)

        # output size: [batch_size, 1, decode_len]
        output = self.wavenet.predict_sequence(forward_input)

        return output.squeeze(1)
