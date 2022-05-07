import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder


def scale_numericals(feat, method='std'):
    assert method in ['std', 'min_max']
    if method == 'std':
        scaler = StandardScaler()
    elif method == 'min_max':
        scaler = MinMaxScaler()
        
    scaler.fit(feat)
    scaled_feat = scaler.transform(feat)
    feat_df = pd.DataFrame(scaled_feat, columns=feat.columns, index=feat.index)
    return feat_df

def encode_categoricals(feat, method='one_hot'):
    encoder = OneHotEncoder() # NaN     handle_unknown='ignore'
    encoder.fit(feat)
    encoded_feat = encoder.transform(feat).toarray() # from spmatrix to array
    encoded_df = pd.DataFrame(encoded_feat, columns=encoder.get_feature_names(feat.columns), index=feat.index)
    return encoded_df


def process_timestampe_numericals(ts):
    df = pd.DataFrame(index=ts.index)
    # 月份
    df['月']=ts.dt.month
    # 日
    df['日']=ts.dt.day
    # 小时
    df['时']=ts.dt.hour
    # 一天中的第几分钟
    df['一天中的第几分钟']=ts.dt.hour * 60 + ts.dt.minute
    # 星期几；
    df['星期几']=ts.dt.dayofweek
    # 一月中的第几天
    df['一个月的第几天'] = ts.dt.days_in_month
    # 一年中的第几天
    df['一年中的第几天']=ts.dt.dayofyear
    # 一年中的第几周
    df['一年中的第几周']=ts.dt.isocalendar().week    
    return df


def process_timestamp_categoricals(ts):
    df = pd.DataFrame(index=ts.index)
    # 是否月初
    df['是否月初'] = ts.dt.is_month_start
    # 是否月末
    df['是否月末'] = ts.dt.is_month_end
    # 一天中哪个时间段：凌晨、早晨、上午、中午、下午、傍晚、晚上、深夜；
    period_dict ={
        23: '深夜', 0: '深夜', 1: '深夜',
        2: '凌晨', 3: '凌晨', 4: '凌晨',
        5: '早晨', 6: '早晨', 7: '早晨',
        8: '上午', 9: '上午', 10: '上午', 11: '上午',
        12: '中午', 13: '中午',
        14: '下午', 15: '下午', 16: '下午', 17: '下午',
        18: '傍晚',
        19: '晚上', 20: '晚上', 21: '晚上', 22: '晚上',
    }
    df['时间段']=ts.dt.hour.map(period_dict)
    df['星期几']=ts.dt.dayofweek
    return df


def process_timestamp(ts):
    ts_num = process_timestampe_numericals(ts)
    ts_cate = process_timestamp_categoricals(ts)
    return ts_num, ts_cate