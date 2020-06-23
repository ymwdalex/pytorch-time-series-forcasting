from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import math

import torch
from torch.utils.data import Dataset

import logging
logging.basicConfig(level=logging.DEBUG)

# global variables
LEN_ALL_TS = 1970
LEN_DECODE = 28


def read_data(input_dir):
    '''
    Read dataframe from CSV files
    '''
    logging.info("Reading files...")

    calendar = pd.read_csv(f"{input_dir}/calendar.csv")
    sell_prices = pd.read_csv(f"{input_dir}/sell_prices.csv")
    sales_train_val = pd.read_csv(f"{input_dir}/sales_train_validation.csv")
    submission = pd.read_csv(f"{input_dir}/sample_submission.csv")

    print("Calendar shape:", calendar.shape)
    print("Sell prices shape:", sell_prices.shape)
    print("Sales train shape:", sales_train_val.shape)
    print("Submission shape:", submission.shape)

    return calendar, sell_prices, sales_train_val, submission


def process_sale_data(df):
    '''
    process sales data:
    - Convert string type categorical features to ordinal features
    '''
    cat_cols = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ]

    encoder = OrdinalEncoder()
    df[cat_cols] = encoder.fit_transform(df[cat_cols])
    return df


def process_calendar(calendar):
    '''
    process calendar data
    - add week number
    - Fill NA
    - ordinal encoding categarical features
    - transpose the dataframe
    '''

    # add week number column
    calendar['date'] = calendar['date'].apply(pd.to_datetime)
    calendar['week_number'] = calendar['date'].dt.week

    # fill NA
    nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in nan_features:
        calendar[feature].fillna('unknown', inplace=True)

    # ordianl encoding
    calendar_select_cols = ['wday',
                            'month',
                            'year',
                            'week_number',
                            'event_name_1',
                            'event_type_1',
                            'event_name_2',
                            'event_type_2',
                            'snap_CA',
                            'snap_TX',
                            'snap_WI']
    encoder = OrdinalEncoder()
    calendar[calendar_select_cols] = encoder.fit_transform(calendar[calendar_select_cols])

    # transpose the dataframe
    # - row: categorical features
    # - columns: d1 - d1969
    days_range = range(1, LEN_ALL_TS)
    time_series_columns = [f'd_{i}' for i in days_range]
    calendar_df_t = pd.DataFrame(calendar[calendar_select_cols].values.T,
                                 index=calendar_select_cols,
                                 columns=time_series_columns)
    calendar_df_t = calendar_df_t.fillna(0)

    return calendar_df_t


def get_encode_decode_data(encode_start,
                           encode_end,
                           decode_start,
                           decode_end,
                           sales_train,
                           calendar,
                           processed_calendar,
                           categorical_feat_l,
                           xdaysago=[365, 91],
                           is_pred=False):
    '''
    Generate encode and decode data

    Input
        encode_start, encode_end:
            - date range for encoding time series
            - type: string
        decode_start, decode_end
            - date range for decoding time series
            - type: string
        sales_train
            - sale data
            - type: dataframe
        calendar
            - calendar data
            - type: dataframe
        processed_calendar
            - processed calendar data
            - type: dataframe
        categorical_feat_l
            - categorical feature used by the model
            - type: list
        is_pred=False
            - prediction set or not. If prediction set, the decoder data is None
        xdaysago=[365, 91]
            - history feature
            - type: list
    Return
        ts_encode
            - time series for encode
            - type: numpy
            - shape: [nr_ts, len_encode]
        ts_decode
            - time series for decode
            - type: numpy
            - shape: [nr_ts, len_decode]
        ts_xdaysago_encode
            - history data for encode
            - type: numpy
            - shape: [nr_ts, len_encode, len_xdaysago_list]
        ts_xdaysago_decode
            - history data for decode
            - type: numpy
            - shape: [nr_ts, len_decode, len_xdaysago_list]
        feat_encode
            - categorical feature for encode
            - type: numpy
            - shape: [len_categorical_feature_list, len_encode]
        feat_decode
            - categarical feature for decode
            - type: numpy
            - shape: [len_categorical_feature_list, len_decode]
    '''

    def get_cols_func(cols):
        return [f'd_{i}' for i in cols]

    def get_d_func(d_str):
        return int(calendar[calendar.date == d_str].d.values[0][2:])

    def get_cols_xdaysago(cols, days):
        return ['d_{}'.format(i-days) for i in cols]

    def get_cols_xdaysago_p1(cols, days):
        return ['d_{}'.format(i-days+1) for i in cols]

    def get_cols_xdaysago_m1(cols, days):
        return ['d_{}'.format(i-days-1) for i in cols]

    def ts_xdays_ago(start, end, days_l):
        ts_l = []
        for days in days_l:
            col_range = range(get_d_func(start), get_d_func(end)+1)
            col_xdaysago = get_cols_xdaysago(col_range, days)
            col_xdaysago_p1 = get_cols_xdaysago_p1(col_range, days)
            col_xdaysago_m1 = get_cols_xdaysago_m1(col_range, days)

            # average [-1, 0 , +1] day histroy data
            ts = (sales_train[col_xdaysago].values +
                  sales_train[col_xdaysago_p1].values +
                  sales_train[col_xdaysago_m1].values)/3

            # reshape ts to 3 dimension [nr_ts, len_en(de)code, 1]
            ts_l.append(np.expand_dims(ts, axis=2))

        return np.concatenate(ts_l, axis=2)

    def daym1(day_str):
        theday = datetime.strptime(day_str, "%Y-%m-%d")
        prevday = theday + timedelta(days=1)
        return datetime.strftime(prevday, "%Y-%m-%d")

    # ---- encoder part --------
    col_encode = get_cols_func(range(get_d_func(encode_start), get_d_func(encode_end)+1))
    ts_encode = sales_train[col_encode].values
    if xdaysago is not None:
        ts_xdaysago_encode = ts_xdays_ago(encode_start, encode_end, xdaysago)
    else:
        ts_xdaysago_encode = None

    # N.B. feature need to shift 1!!!
    feat_col_encode = get_cols_func(range(get_d_func(daym1(encode_start)), get_d_func(daym1(encode_end))+1))
    feat_encode = processed_calendar.loc[categorical_feat_l][feat_col_encode].values if categorical_feat_l is not None else None
    # ------------------------

    # ----- decoder part ------
    col_decode = get_cols_func(range(get_d_func(decode_start), get_d_func(decode_end)+1))
    if xdaysago is not None:
        ts_xdaysago_decode = ts_xdays_ago(decode_start, decode_end, xdaysago)
    else:
        ts_xdaysago_decode = None

    # N.B. feature need to shift 1!!!
    feat_col_decode = get_cols_func(range(get_d_func(daym1(decode_start)), get_d_func(daym1(decode_end))+1))
    feat_decode = processed_calendar.loc[categorical_feat_l][feat_col_decode].values if categorical_feat_l is not None else None
    # --------------------------

    if not is_pred:
        ts_decode = sales_train[col_decode].values
    else:
        ts_decode = np.zeros((len(ts_encode), LEN_DECODE))

    return ts_encode, ts_decode, ts_xdaysago_encode, ts_xdaysago_decode, feat_encode, feat_decode


def get_fixed_feat(fixed_feat_l, sales_train):
    if fixed_feat_l is None:
        return None
    else:
        return sales_train[fixed_feat_l].values


class TSDataset(Dataset):
    def __init__(self,
                 device,
                 ts_encode,
                 ts_decode,
                 ts_xdaysago_encode,
                 ts_xdaysago_decode,
                 feat_encode,
                 feat_decode,
                 fixed_feat):
        '''
        PyTorch dataset class for M5 data

        Input:
            device: 
                - GPU or CPU
                - type: pytorch device
            ts_encode
                - time series for encode
                - type: numpy
                - shape: [nr_ts, len_encode]
            ts_decode
                - time series for decode
                - type: numpy
                - shape: [nr_ts, len_decode]
            ts_xdaysago_encode
                - history data for encode
                - type: numpy
                - shape: [nr_ts, len_encode, len_xdaysago_list]
            ts_xdaysago_decode
                - history data for decode
                - type: numpy
                - shape: [nr_ts, len_decode, len_xdaysago_list]
            feat_encode
                - categorical feature for encode
                - type: numpy
                - shape: [len_categorical_feature_list, len_encode]
            feat_decode
                - categarical feature for decode
                - type: numpy
                - shape: [len_categorical_feature_list, len_decode]
            fixed_feat
                - fixed feature
                - type: numpy
                - shape: [nr_ts, len_fixed_feature_list]


        '''

        # number of time series for encode and decode should be the same
        # In M5 data, nr_ts is 30490
        assert(len(ts_encode) == len(ts_decode))

        nr_ts = len(ts_encode)

        # ------- ts en(de)code -----------------
        # ts_en(de)code size: [nr_ts, len_en(de)code]
        # ts_en(de)code_tensor size: [number_ts, len_en(de)code, feature_dimension]
        ts_encode_tensor = torch.FloatTensor(ts_encode).view(nr_ts, -1, 1)
        ts_decode_tensor = torch.FloatTensor(ts_decode).view(nr_ts, -1, 1)

        # normalization
        # time series are normalized row by row independently
        mean_x = torch.mean(ts_encode_tensor, dim=1, keepdims=True)
        std_x = torch.std(ts_encode_tensor, dim=1, keepdims=True)
        std_x[std_x==0] = 1

        ts_encode_tensor_norm = (ts_encode_tensor - mean_x) / std_x
        ts_decode_tensor_norm = (ts_decode_tensor - mean_x) / std_x
        # # only mean
        # ts_encode_tensor_norm = (ts_encode_tensor - mean_x)
        # ts_decode_tensor_norm = (ts_decode_tensor - mean_x)

        self.mean_x = mean_x.to(device)
        self.std_x = std_x.to(device)

        self.ts_encode_norm = ts_encode_tensor_norm.to(device)
        self.ts_decode_norm = ts_decode_tensor_norm.to(device)

        # also retunr original ts_decode tensor for evaluation
        self.ts_decode_true = ts_decode_tensor.to(device)

        # ------- historical data en(de)code -----------------
        if ts_xdaysago_encode is not None and ts_xdaysago_decode is not None:    
            ts_xdaysago_encode = torch.FloatTensor(ts_xdaysago_encode)
            ts_xdaysago_decode = torch.FloatTensor(ts_xdaysago_decode)

            self.ts_xdaysago_encode = ((ts_xdaysago_encode - mean_x) / std_x).to(device)
            self.ts_xdaysago_decode = ((ts_xdaysago_decode - mean_x) / std_x).to(device)
        else:
            self.ts_xdaysago_encode = None
            self.ts_xdaysago_decode = None
        # # only mean
        # self.ts_xdaysago_encode = ((ts_xdaysago_encode - mean_x)).to(device)
        # self.ts_xdaysago_decode = ((ts_xdaysago_decode - mean_x)).to(device)

        # ------- categorical feature data en(de)code -----------------
        if feat_encode is not None and feat_decode is not None:
            # feat_encode size: [feat_encode_dimension, len_en(de)code]
            # cat_feat_en(de)code size: [nr_ts, len_en(de)code, feat_encode_dimension]
            feat_encode_tensor = torch.LongTensor(feat_encode.T)
            self.cat_feat_encode = feat_encode_tensor.repeat((nr_ts, 1, 1)).to(device)

            feat_decode_tensor = torch.LongTensor(feat_decode.T)
            self.cat_feat_decode = feat_decode_tensor.repeat((nr_ts, 1, 1)).to(device)
        else:
            self.cat_feat_encode = None
            self.cat_feat_decode = None

        # ------- fixed feature data en(de)code -----------------
        if fixed_feat is not None:
            # fixed_feat size: [nr_ts, fixed_dim]
            # fixed_feat_tensor size: [nr_ts, fixed_dim]
            fixed_feat_tensor = torch.LongTensor(fixed_feat)
            self.fixed_feat_encode = fixed_feat_tensor.to(device)
        else:
            self.fixed_feat_encode = None

        print("ts_encode_norm shape {}\n".format(self.ts_encode_norm.size()),
              "ts_decode_norm shape {}\n".format(self.ts_decode_norm.size()),
              "ts_decode_true shape {}\n".format(self.ts_decode_true.size()),
              "ts_xdaysago_encode shape {}\n".format(self.ts_xdaysago_encode.size() if self.ts_xdaysago_encode is not None else None),
              "ts_xdaysago_decode shape {}\n".format(self.ts_xdaysago_decode.size() if self.ts_xdaysago_decode is not None else None),
              "cat_feat_encode shape {}\n".format(self.cat_feat_encode.size() if self.cat_feat_encode is not None else None),
              "cat_feat_decode shape {}\n".format(self.cat_feat_decode.size() if self.cat_feat_decode is not None else None),
              "fixed_feat_encode shape {}\n".format(self.fixed_feat_encode.size() if self.fixed_feat_encode is not None else None),
              "mean_x shape {}\n".format(self.mean_x.size()),
              "std_x shape {}\n".format(self.std_x.size()))

    def __len__(self):
        return len(self.ts_encode_norm)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.ts_encode_norm[idx],
                self.ts_decode_norm[idx],
                self.ts_decode_true[idx],
                self.mean_x[idx],
                self.std_x[idx],
                self.ts_xdaysago_encode[idx],
                self.ts_xdaysago_decode[idx],
                self.cat_feat_encode[idx] if self.cat_feat_encode is not None else None,
                self.cat_feat_decode[idx] if self.cat_feat_decode is not None else None,
                self.fixed_feat_encode[idx] if self.fixed_feat_encode is not None else None)


def get_cat_feat_emb_para(categorical_feat_l, processed_calendar):
    '''
    Output:
        cat_emb_para_l: a list of tuple (name_cat_feat, nr_unique, nr_emb_dim)
            - name_cat_feat: the name of categorical feature
            - nr_unique: number of unique values of the categorical feature, input dim of embedding
            - nr_emb_dim: output dimemsion of embedding
    '''
    if categorical_feat_l is None:
        return None

    rtv = []
    for cat_feat in categorical_feat_l:
        nr_unique = processed_calendar.loc[cat_feat].nunique()
        nr_emb_dim = math.ceil(math.log10(nr_unique))
        rtv.append((cat_feat, nr_unique, nr_emb_dim))
    return rtv


def get_fixed_feat_emb_para(fixed_feat_l, sale_train):
    '''
    Output:
        fixed_emb_para_l: a list of tuple (name_fixed_feat, nr_unique, nr_emb_dim)
                - name_fixed_feat: the name of fixed feature
                - nr_unique: number of unique values of the categorical feature, input dim of embedding
                - nr_emb_dim: output dimemsion of embedding    
    '''
    if fixed_feat_l is None:
        return None

    rtv = []
    for fixed_feat in fixed_feat_l:
        nr_unique = sale_train[fixed_feat].nunique()
        nr_emb_dim = math.ceil(math.log10(nr_unique)) + 1
        rtv.append((fixed_feat, nr_unique, nr_emb_dim))
    return rtv


def get_dataset(input_dir,
                device,
                train_encode_decode_boundray,
                val_encode_decode_boundray,
                categorical_feat_l,
                fixed_feat_l,
                xdaysago):
    '''
    Build a PyTorch dataset for M5 data

    Input
        input_dir
            - folder where M5 data locates
        device
            - PyTorch device, cpu or gpu
        train_encode_decode_boundray
            - tuple of (encode_start_date, encode_end_date, decode_start_date, decode_end_date) 
        val_encode_decode_boundray
            - tuple of (encode_start_date, encode_end_date, decode_start_date, decode_end_date) 
        categorical_feat_l
            - list of categorical features used by model
        fixed_feat_l
            - list of fixed feature used by model
        xdaysago
            - list of date for history data

    Output:
        dataloader instance
    '''

    calendar, sell_prices, sales_train, submission = read_data(input_dir)
    sales_train = process_sale_data(sales_train)
    processed_calendar = process_calendar(calendar)

    train_tuple = get_encode_decode_data(*train_encode_decode_boundray,
                                         sales_train,
                                         calendar,
                                         processed_calendar,
                                         categorical_feat_l,
                                         xdaysago=xdaysago,
                                         is_pred=False)

    val_tuple = get_encode_decode_data(*val_encode_decode_boundray,
                                       sales_train,
                                       calendar,
                                       processed_calendar,
                                       categorical_feat_l,
                                       xdaysago=xdaysago,
                                       is_pred=False)

    fixed_feat = get_fixed_feat(fixed_feat_l, sales_train)

    dataset_train = TSDataset(device, *train_tuple, fixed_feat)

    dataset_val = TSDataset(device, *val_tuple, fixed_feat)

    return dataset_train, dataset_val
