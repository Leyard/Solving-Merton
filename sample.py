import numpy as np
from scipy.stats import norm


colnames = ["ID", "ASSET", "DEBT", "RISKFREE", "ASSETVOL", "MATURITY", 
            "EQUITY", "EQUITYVOL"]


def black_scholes(asset, debt, rf_rate, asset_vol, maturity):
    d1 = np.log(asset/debt) - (rf_rate+0.5*asset_vol*asset_vol)*maturity
    d1 = d1 / (asset_vol*np.sqrt(maturity))
    d2 = d1 - asset_vol*np.sqrt(maturity)
    call = asset*norm.cdf(d1) - debt*np.exp(-rf_rate*maturity)*norm.cdf(d2)
    return call


def equity_vol(asset, debt, equity, rf_rate, asset_vol, maturity):
    d1 = np.log(asset/debt) - (rf_rate+0.5*asset_vol*asset_vol)*maturity
    eq_vol = asset_vol/equity*norm.cdf(d1)*asset
    return eq_vol


def sameple_generator():
    asset = np.random.uniform(1e7, 1e12)
    debt = asset*np.random.uniform()
    rf_rate = 0.028
    asset_vol = np.random.uniform(0.10, 0.80)
    maturity = np.random.uniform(0.01, 10)
    equity = black_scholes(asset, debt, rf_rate, asset_vol, maturity)
    eq_vol = equity_vol(asset, debt, equity, rf_rate, asset_vol, maturity)
    return [asset, debt, rf_rate, asset_vol, maturity, equity, eq_vol]


wf = open("dataset_train.txt", "w")
wf.write(",".join(colnames) + "\n")
for i in range(10000):
    obs = [i+1] + sameple_generator()
    wf.write(",".join(list(map(str, obs))) + "\n")
wf.close()


wf = open("dataset_test.txt", "w")
wf.write(",".join(colnames) + "\n")
for i in range(2000):
    obs = [i+1] + sameple_generator()
    wf.write(",".join(list(map(str, obs))) + "\n")
wf.close()
