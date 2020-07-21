#!/usr/bin/env python
# coding: utf-8

import argparse

import numpy as np
import pandas as pd

from tsfeatures.tsfeatures_r import tsfeatures_r_wide
from tscompdata import tourism
from rpy2.robjects.packages import importr
from fforma.meta_model import MetaModels
from fforma.base_models import Naive2
from fforma.base_models_r import (
    ARIMA,
    ETS,
    NNETAR,
    TBATS,
    STLM,
    STLMFFORMA,
    RandomWalk,
    ThetaF,
    Naive,
    SeasonalNaive
)

def split_holdout(series):
    h = series['horizon']
    y = series['y']

    y_train, y_val = y[:-h], y[-h:]

    return y_train, y_val

def ds_holdout(series):
    h = series['horizon']
    ds = np.arange(1, h + 1)

    return ds

def main(args):
    directory = args.directory

    tourism_data = tourism.get_complete_wide_data(directory)

    y_train_val = [split_holdout(row) for idx, row in tourism_data.iterrows()]
    tourism_data['y_train'], tourism_data['y_val'] = zip(*y_train_val)

    tourism_data['ds'] = [ds_holdout(row) for idx, row in tourism_data.iterrows()]

    stats = importr('stats')

    meta_models = {
         'auto_arima_forec': lambda freq: ARIMA(freq, stepwise=False, approximation=False),
         'ets_forec': ETS,
         'nnetar_forec': NNETAR,
         'tbats_forec': TBATS,
         'stlm_ar_forec': lambda freq: STLMFFORMA(freq, modelfunction=stats.ar) if freq > 1 else ARIMA(freq, d=0, D=0),
         'rw_drift_forec': lambda freq: RandomWalk(freq=freq, drift=True),
         'theta_forec': ThetaF,
         'naive_forec': Naive,
         'snaive_forec': SeasonalNaive,
         'y_hat_naive2': Naive2,
    }

    print('Validation meta data')

    validation_data = tourism_data[['unique_id', 'ds', 'horizon', 'seasonality', 'y_train', 'y_val']].rename(columns={'y_train': 'y'})
    vaidation_models = MetaModels(meta_models).fit(validation_data)
    validation_preds = vaidation_models.predict(validation_data.drop(['seasonality', 'y'], 1))

    validation_features = tsfeatures_r_wide(validation_data, parallel=True).reset_index()

    print('Test meta data')

    test_data = tourism_data[['unique_id', 'ds', 'horizon', 'seasonality', 'y', 'y_test']]
    test_models = MetaModels(meta_models).fit(test_data)
    test_preds = test_models.predict(test_data.drop(['seasonality', 'y'], 1))

    test_features = tsfeatures_r_wide(test_data, parallel=True).reset_index()

    save_data = (validation_features, validation_preds, test_features, test_preds)

    pd.to_pickle(save_data, directory + '/tourism-meta-data.pickle')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get metadata for tourism')
    parser.add_argument("--directory", required=True, type=str,
                      help="directory where tourism data will be downloaded")

    args = parser.parse_args()

    main(args)
