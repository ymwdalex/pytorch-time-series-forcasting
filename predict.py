import torch
import numpy as np
import utils


def predict(model, dataloader, len_encode, len_decode, device):
    n_obs = len(dataloader.dataset)
    batch_sz = dataloader.batch_size

    pred = np.zeros((n_obs, len_decode))
    pred_orig = np.zeros((n_obs, len_decode))

    currently_deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True

    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (src, trg, _, mean, std, src_xdaysago, trg_xdaysago, cat_encode, cat_decode, fixed_encode) in enumerate(dataloader):
            start_index = i * batch_sz
            end_index = min(start_index + batch_sz, n_obs)

            # src dim: [batch_size, len_ts, 1] --> [len_encode, batch_size, 1]
            # trg dim: [batch_size, len_ts, 1] --> [len_decode, batch_size, 1]
            # src_xdaysago dim: [batch_size, len_ts, historical_data_dim] --> [len_decode, batch_size, historical_data_dim]
            # trg_xdaysago dim: [batch_size, len_ts, historical_data_dim] --> [len_decode, batch_size, historical_data_dim]
            # cat_encode dim: [batch_size, len_ts, cat_feature_emb_dim] --> [len_decode, batch_size, cat_feature_emb_dim]
            # fixed_encode dim: [batch_size, len_ts, fixed_feature_emb_dim] --> [len_decode, batch_size, fixed_feature_emb_dim]
            src = src.permute(1, 0, 2)
            trg = trg.permute(1, 0, 2)
            src_xdaysago = src_xdaysago.permute(1, 0, 2)
            trg_xdaysago = trg_xdaysago.permute(1, 0, 2)
            cat_encode = cat_encode.permute(1, 0, 2)
            cat_decode = cat_decode.permute(1, 0, 2)

            # turn off teacher forcing
            output = model(src, trg, src_xdaysago, trg_xdaysago, cat_encode, cat_decode, fixed_encode, 0)
            output = output.squeeze().permute(1, 0)

            # append the prediction batch by batch
            pred[start_index:end_index, :] += utils.to_numpy(output)

            # mean size: [batch_size, 1, 1] --> [batch_size, 1]
            std = std.squeeze(dim=2)
            mean = mean.squeeze(dim=2)

            output_inverse = (output * std) + mean

            # # only mean
            # output_inverse = output + mean
            pred_orig[start_index:end_index, :] += utils.to_numpy(output_inverse)

    torch.backends.cudnn.deterministic = currently_deterministic

    return pred, pred_orig
