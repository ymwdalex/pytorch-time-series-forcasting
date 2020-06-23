import torch
from torch import nn
import numpy as np
import pandas as pd
from predict import predict


def init_weights(m):
    for name, param in m.named_parameters():
        if param.data.dim() > 1:
            print(f'{name} with size {param.data.shape}: orthogonal_')
            nn.init.orthogonal_(param.data)
        else:
            print(f'{name} with size {param.data.shape}: uniform_')
            nn.init.uniform_(param.data, -0.05, 0.05)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class RMSELoss(nn.Module):
    '''
    Use a nn class for RMSE loss
    '''
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def to_cpu(x):
    return x.contiguous().detach().cpu()


def to_numpy(x):
    return to_cpu(x).numpy()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, epoch, lr):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} in epoch {epoch} lr {lr:.6f}; \n\t\tLoss {val_loss:.6f}; Best score {self.best_score:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,  epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'cnn.pt')
        self.val_loss_min = val_loss


def vis(dataloader_vis, model, len_encode, len_decode, device):
    pred, pred_orig = predict(model, dataloader_vis, len_encode, len_decode, device)  
    print('pred shape', pred.shape)
    
    idx = np.random.randint(len(dataloader_vis.dataset))
    print('Visualize time series [{}]'.format(idx))
    
    decode_norm = dataloader_vis.dataset[idx][1].view(len_decode).numpy()
    pred_norm = pred[idx]

    encode_norm = dataloader_vis.dataset[idx][0].view(len_encode).numpy()
    
    array_nan_encode = np.empty(len_encode)
    array_nan_encode[:] = np.nan
    array_nan_decode = np.empty(len_decode)
    array_nan_decode[:] = np.nan
    
    encode_array = np.concatenate([encode_norm, array_nan_decode])
    decode_array = np.concatenate([array_nan_encode, decode_norm])
    pred_array = np.concatenate([array_nan_encode, pred_norm])

    df = pd.DataFrame({'encode': encode_array,
                       'pred': pred_array, 
                       'decode': decode_array})
    df.plot()

    return pred_norm, decode_norm