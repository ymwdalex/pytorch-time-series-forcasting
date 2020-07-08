import random
import torch
import torch.nn as nn
from embedding import CategoricalEmbedding, Conv1DLayer, FCLayer, FixedFeatEmbedding


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """
        input_size
            – The number of expected features in the input x
        hidden_size
            – The number of features in the hidden state h
        num_layers
            – Number of recurrent layers.
            - setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU
        dropout
            – If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer,
              with dropout probability equal to dropout. Default: 0
        https://pytorch.org/docs/stable/nn.html#torch.nn.GRU
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, input):
        # input dim: [encode_len, batch_size, encode_feat_dim]
        # h_0: since nothing provided, default to zero

        # output dim: [seq_len, batch, num_directions * hidden_size], we don't use it
        # hidden dim: [num_layers*num_directions, batch_size, hidden_dim]
        output, hidden = self.gru(input)

        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """
        input_size
            – The number of expected features in the input x
        hidden_size
            – The number of features in the hidden state h
        num_layers
            – Number of recurrent layers
            - setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU
        dropout
            – If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer,
              with dropout probability equal to dropout. Default: 0
        https://pytorch.org/docs/stable/nn.html#torch.nn.GRU
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, input, hidden):
        # input dim: [batch_size, decode_feat_dim]
        # hidden dim: [num_layers * num_directions, batch_size, hidden_dim]

        # input dim: [1, batch_size, decode_feat_dim]
        input = input.unsqueeze(0)

        # output dim: [1, batch_size, hidden_size]
        # hidden dim: [(num_layers * num_directions, batch_size, hidden_dim]
        output, hidden = self.gru(input, hidden)

        return output, hidden


class ContextEnhanceLayer(nn.Module):
    def __init__(self, context_in_dim, context_out_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(context_in_dim, context_out_dim, bias=False), nn.Tanh()
        )

    def forward(self, input):
        out = self.classifier(input)
        return out


class Seq2Seq(nn.Module):
    def __init__(
        self,
        device,
        hid_dim_encode,
        hid_dim_decode,
        n_layers_rnn,
        dropout_rnn,
        categorical_feat_emb_lookup,
        fixed_feat_emb_lookup,
        xdaysago,
        fc1_out_dim,
        fc2_out_dim,
        conv1d_output_dim,
        conv1d_kernal_size,
    ):
        """
        encoder/decoder parameter
            - hid_dim_encode: The number of features in the hidden state of encoder
            - hid_dim_decode: The number of features in the hidden state of decoder
            - n_layers_rnn: nr of layer in GRU
            - dropout_rnn: dropout rate in GRU
        categorical feature parameter
            - categorical_feat_emb_lookup: (name_cat_feat, nr_unique, nr_emb_dim)
        fixed feature parameter
            - fixed_feat_emb_lookup: (name_cat_feat, nr_unique, nr_emb_dim)
        history data parameter
            - xdaysago: list of historical date, e.g., [365, 91]
        fully connected layer parameter
            - fc_in_dim, fc1_out_dim, fc2_out_dim
            - fully connected layer as output
            - here we use 2 layers
        CONV1D layer parameter
            - conv1d_output_dim
            - conv1d_kernal_size
        """
        super().__init__()

        self.cat_emb_layer = CategoricalEmbedding(categorical_feat_emb_lookup).to(
            device
        )
        self.fixed_emb_layer = FixedFeatEmbedding(fixed_feat_emb_lookup).to(device)
        self.fc_layer = FCLayer(hid_dim_decode, fc1_out_dim, fc2_out_dim).to(device)
        self.context_layer = ContextEnhanceLayer(hid_dim_encode, hid_dim_encode).to(
            device
        )
        self.seq_conv_layer = Conv1DLayer(conv1d_output_dim, conv1d_kernal_size).to(
            device
        )

        input_dim_encode = (
            1
            + sum([cat[2] for cat in categorical_feat_emb_lookup])
            + sum([fixed_feat[2] for fixed_feat in fixed_feat_emb_lookup])
            + len(xdaysago)
            + conv1d_output_dim
        )
        self.encoder = EncoderRNN(
            input_dim_encode, hid_dim_encode, n_layers_rnn, dropout_rnn
        ).to(device)

        # some features are not used in decode, such as conv1d result
        input_dim_decode = (
            1
            + sum([cat[2] for cat in categorical_feat_emb_lookup])
            + sum([fixed_feat[2] for fixed_feat in fixed_feat_emb_lookup])
            + len(xdaysago)
        )
        self.decoder = DecoderRNN(
            input_dim_decode, hid_dim_decode, n_layers_rnn, dropout_rnn
        ).to(device)

        self.device = device

        assert (
            self.encoder.num_layers == self.decoder.num_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(
        self,
        src_ts,
        trg_ts,
        src_xdaysago,
        trg_xdaysago,
        cat_encode,
        cat_decode,
        fixed_feat,
        teacher_forcing_ratio=0.5,
    ):
        """
        src_ts size: [encode_len, batch_size, 1]
        trg_ts size: [decode_len, batch_size, 1]
        src_xdaysago size: [encode_len, batch_size, history_data_dim]
        trg_xdaysago size: [decode_len, batch_size, history_data_dim]
        cat_encode size: [encode_len, batch_size, cat_feat_dim]
        cat_decode size: [decode_len, batch_size, cat_feat_dim]
        fixed_feat size: [batch_size, fixed_feat_dim]
        """

        # tensor to store decoder outputs
        encode_len = src_ts.shape[0]
        decode_len = trg_ts.shape[0]
        batch_size = trg_ts.shape[1]
        trg_dim = trg_ts.shape[2]

        # initial outputs size: [decoder_len, batch_size, 1]
        outputs = torch.zeros(decode_len, batch_size, trg_dim).to(self.device)

        # categorical feature embedding size: [en(de)code_len, batch_size, cat_feat_emb_dim]
        # cat_feat_emb_dim = sum([cat[2] for cat in categorical_feat_emb_lookup])
        cat_encode_emb = self.cat_emb_layer(cat_encode).to(self.device)
        cat_decode_emb = self.cat_emb_layer(cat_decode).to(self.device)

        # fixed feature embedding size: [batch_size, fixed_feat_emb_dim]
        # fixed_feat_emb_dim = sum([fixed[2] for fixed in fixed_feat_emb_lookup])
        fixed_emb = self.fixed_emb_layer(fixed_feat).to(self.device)

        # repeat fixed embedding feature to [endcode_len, batch_size, fixed_feat_emb_dim]
        fixed_emb = fixed_emb.repeat(encode_len, 1, 1)

        # get conv1d embadding, conv1d size: [encode_len, batch_size, conv_output_dim]
        conv1d_emb = self.seq_conv_layer(src_ts)

        # encode_input size: [encode_len, batch_size, (ts_dim + conv1d_emb + xdaysago_dim + cat_emb_dim + fixed_emb_dim)]
        encode_input = torch.cat(
            [src_ts, conv1d_emb, src_xdaysago, cat_encode_emb, fixed_emb], 2
        )

        # we don't use conv1d feature as decoder initial input
        encode_input_for_first_decode = torch.cat(
            [src_ts, src_xdaysago, cat_encode_emb, fixed_emb], 2
        )

        # Use encoder hidden state output as the context vector
        # context dim: [(num_layers * num_directions, batch_size, encoder_hidden_dim]
        _, context = self.encoder(encode_input)

        # enhance context vector by one Linear + Tahn layer
        # context dim: [(num_layers * num_directions, batch_size, encoder_hidden_dim]
        context = self.context_layer(context)
        hidden = context

        # ----------------
        # besides the decoder input, context can also be used as final dense layer
        # we do not test it in this code
        # -------------------

        # use the last timestamp of encode as the input of decode
        # size [batch_size, ts_dim + ts_daysago_dim + cat_emd_dim + fixed_emb_dim]
        decode_input = encode_input_for_first_decode[-1]

        # predict step-by-step
        for t in range(decode_len):
            # receive output tensor (predictions) as the new decode_input
            # decode_input dim: [batch_size, ts_dim + ts_daysago_dim + cat_emd_dim + fixed_emb_dim]
            # output dim: [seq_len (1), batch_size, decode_hidden_dim]
            # hidden dim: [(num_layers * num_directions, batch_size, decode_hidden_dim]
            output, hidden = self.decoder(decode_input, hidden)

            # fc layer input dim: [batch_size, decode_feat_dim]
            # final_output dim: [batch_size, 1]
            final_output = self.fc_layer(output.squeeze(0))

            # place predictions in a tensor holding predictions for each token
            outputs[t] = final_output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next time series value as next input
            # if not, use predicted time series value
            if teacher_force:
                decode_input = torch.cat(
                    [trg_ts[t], trg_xdaysago[t], cat_decode_emb[t], fixed_emb[t]], 1
                )
            else:
                decode_input = torch.cat(
                    [final_output, trg_xdaysago[t], cat_decode_emb[t], fixed_emb[t]], 1
                )

        return outputs
