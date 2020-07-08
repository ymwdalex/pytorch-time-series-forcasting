import torch
from torch import nn


def train(
    model, dataloader, optimizer, scheduler, criterion, clip, forch_teaching_rate=0.5
):
    model.train()

    epoch_loss = 0

    for (
        batch_idx,
        (
            src,
            trg,
            _,
            _,
            _,
            src_xdaysago,
            trg_xdaysago,
            cat_encode,
            cat_decode,
            fixed_encode,
        ),
    ) in enumerate(dataloader):

        optimizer.zero_grad()

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

        # output dim: [len_decode, batch_size, 1]
        output = model(
            src,
            trg,
            src_xdaysago,
            trg_xdaysago,
            cat_encode,
            cat_decode,
            fixed_encode,
            forch_teaching_rate,
        )

        loss = criterion(output, trg)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion):

    model.eval()

    epoch_loss = 0
    epoch_loss_orig = 0

    with torch.no_grad():

        for (
            batch_idx,
            (
                src,
                trg,
                trg_true,
                mean,
                std,
                src_xdaysago,
                trg_xdaysago,
                cat_encode,
                cat_decode,
                fixed_encode,
            ),
        ) in enumerate(dataloader):

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

            # output dim: [len_decode, batch_size, 1]
            # must turn off teacher forcing
            output = model(
                src,
                trg,
                src_xdaysago,
                trg_xdaysago,
                cat_encode,
                cat_decode,
                fixed_encode,
                0,
            )

            loss = criterion(output, trg)
            epoch_loss += loss.item()

            # mean std dim: [batch_size, 1, decode_feat_dim]
            std = std.squeeze(dim=2)
            mean = mean.squeeze(dim=2)

            output_inverse = (output * std) + mean
            # # only mean
            # output_inverse = output + mean
            # output_inverse dim: [len_decode, batch_size, decode_feat_dim]

            trg_true = trg_true.permute(1, 0, 2)
            loss = criterion(output_inverse, trg_true)
            epoch_loss_orig += loss.item()

    return epoch_loss / len(dataloader), epoch_loss_orig / len(dataloader)
