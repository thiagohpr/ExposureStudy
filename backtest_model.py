# %%
# trend following: o comportamento visto ontem vai se perpetuar para hoje

# tese: retornos passados explicam retornos futuros
# tese: os retornos do ibovespa apresentam auto-correlação

# estratégia:
# ret_d-1 > 0 -> compro d0
# ret_d-1 = 0 -> n faço nada
# ret_d-1 < 0 -> vendo d0


# %%
import pandas as pd
import numpy as np

bench = pd.read_csv('20230224_dao_benchmarks.csv')
bench = bench.rename({"Close|adj by CA's|orig currency|IBOV":"IBOV"}, axis = 1)
bench = bench[['IBOV']].replace('-', np.NaN)

bench = bench.rename({"Close|adj by CA's|orig currency|DOLOF":"DOLBRL"}, axis = 1)
bench = bench[['DOLBRL']].replace('-', np.NaN)

ibov = bench[['IBOV', 'DOLBRL']].dropna()
ibov['IBOV'] = pd.to_numeric(ibov['IBOV'])
ibov['DOLBRL'] = pd.to_numeric(ibov['DOLBRL'])


# %% 

def estrategia_simples(ser: pd.Series) -> pd.Series:
    condlist = [
        ser > 0,
        ser == 0,
        ser < 0,
    ]
    choicelist = [-1, 0, 1]
    result = np.select(condlist, choicelist)
    return result

def estrategia_continua(ser: pd.Series) -> pd.Series:
    starting_pos = 1
    return starting_pos - 3*ser
# %%
ibov['pct'] = ibov['IBOV'].pct_change()
# ibov['estrat'] = estrategia_simples(ibov['pct'])
# ibov['cont'] = estrategia_continua(ibov['pct'])
# ibov
# %%
ibov['pos'] = ibov['cont'].shift(2)
ibov['pnl'] = ibov['pct'] * ibov['pos']
ibov

# %%
from multifac import statistics
print(statistics.summary_statistics(ibov['pnl']))
# statistics.max_drawdown(np.cumprod(1+ibov['pnl']))
np.cumprod(1+ibov['pnl']).plot()

# %%
import plotly.express as px
compare_df = bench[['IBOV']].join(np.cumprod(1+ibov['pnl'])).dropna()
compare_df['IBOV'] /= 16106.00
compare_df['pnl'] /= 0.993080
px.line(compare_df, log_y=True)

# %%
import statsmodels.api as sm
sm.graphics.tsa.plot_acf(ibov['pnl'].dropna())