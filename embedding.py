import torch
import torch.nn as nn


class CategoricalEmbedding(nn.Module):
    """
    Embedding layer for categorical features
    """

    def __init__(self, cat_emb_para_l):
        """
        cat_emb_para_l: a list of tuple (name_cat_feat, nr_unique, nr_emb_dim)
            - name_cat_feat: the name of categorical feature
            - nr_unique: number of unique values of the categorical feature, input dim of embedding
            - nr_emb_dim: output dimemsion of embedding
        """
        super().__init__()
        self.cat_emb_layer_l = nn.ModuleList(
            [
                nn.Embedding(cat_emb_para[1], cat_emb_para[2])
                for cat_emb_para in cat_emb_para_l
            ]
        )

    def forward(self, cat_tensor):
        """
        Input: categorical features
            - shape: [en(de)code_len, batch_size, nr_cat_features]
        Output: categorical embedded features
            - shape: [en(de)code_len, batch_size, cat_feature_dim]
        """

        cat_emb_l = []
        for idx in range(len(self.cat_emb_layer_l)):
            # emb_in size: [en(de)code_len, batch_size]
            emb_in = cat_tensor[:, :, idx]

            # emb_out size: [en(de)code_len, batch_size, emb_dim]
            emb_out = self.cat_emb_layer_l[idx](emb_in)

            cat_emb_l.append(emb_out)

        # cat_tensor size: [en(de)code_len, batch_size, cat_feat_dim]
        cat_emb_tensor = torch.cat(cat_emb_l, 2)

        return cat_emb_tensor


class FixedFeatEmbedding(nn.Module):
    """
    Embedding layer for fixed features
    """

    def __init__(self, fixed_emb_para_l):
        """
        fixed_emb_para_l: a list of tuple (name_fixed_feat, nr_unique, nr_emb_dim)
            - name_fixed_feat: the name of fixed feature
            - nr_unique: number of unique values of the categorical feature, input dim of embedding
            - nr_emb_dim: output dimemsion of embedding
        """
        super().__init__()
        # fixed_emb_para_l: [fixed_feature_name, nr_distinct_val, embedding_dim]
        self.fixed_emb_layer_l = nn.ModuleList(
            [
                nn.Embedding(fixed_emb_para[1], fixed_emb_para[2])
                for fixed_emb_para in fixed_emb_para_l
            ]
        )

    def forward(self, fixed_tensor):
        """
        Input: fixed features
            - shape: [en(de)code_len, batch_size, nr_cat_features]
        Output: fixed embedded features
            - shape: [en(de)code_len, batch_size, cat_feature_dim]
        """
        fixed_emb_l = []
        for idx in range(len(self.fixed_emb_layer_l)):
            # emb_in size: [batch_size]
            emb_in = fixed_tensor[:, idx]

            # emb_out size: [batch_size, emb_dim]
            emb_out = self.fixed_emb_layer_l[idx](emb_in)
            fixed_emb_l.append(emb_out)

        # fixed_emb size: [batch_size, all_emb_dim]
        fixed_emb_tensor = torch.cat(fixed_emb_l, 1)
        return fixed_emb_tensor


class FCLayer(nn.Module):
    def __init__(self, fc1_in_dim, fc1_out_dim, fc2_out_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(fc1_in_dim, fc1_out_dim, bias=False),
            nn.ReLU(),
            nn.Linear(fc1_out_dim, 1, bias=False),
        )

    def forward(self, input):
        # input size: [batch_size, decode_feat_dim]
        out = self.classifier(input)
        return out


class Conv1DLayer(nn.Module):
    """
    Conv1d: https://pytorch.org/docs/stable/nn.html#conv1d
    """

    def __init__(self, out_channels, kernel_size):
        """
        out_channels: Number of channels produced by the convolution
        kernel_size: size of the convolving kernel
        """
        super().__init__()

        # since we have 1D time series, the first parameter is always 1
        self.conv1d = nn.Conv1d(
            1,
            out_channels,
            kernel_size,
            padding=int((kernel_size - 1) / 2),
            padding_mode="reflect",
        )

    def forward(self, input):
        # input size: [encode_len, batch_size, 1]
        input = input.permute(1, 2, 0)

        # requirement:
        # - input: batch_size, 1, ts_len
        # - output: batch_size, out_channel, ts_len
        out = self.conv1d(input)

        # output size: ts_len, batch_size, out_channels
        out = out.permute(2, 0, 1)
        return out
