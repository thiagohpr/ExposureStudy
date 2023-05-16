# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime as dt
import plotly.express as px
import warnings
import time
warnings.filterwarnings('ignore')
from IPython.display import clear_output
import yfinance as yf
from scipy import stats
import itertools

# Taxa de juros e S&P
# Outra variável comparativa de carteiras - skillness por ex, blend de variáveis, retorno absoluto, minimizar a maior queda

# %%
all_columns = ['Date','IBOV', 'DOLBRL', 'CDI', 'IBXX', 'SMLL']

bench = pd.read_csv('20230224_dao_benchmarks.csv')
bench = bench.rename({"Close|adj by CA's|orig currency|IBOV":"IBOV","Close|adj by CA's|orig currency|DOLOF":"DOLBRL", "Close|adj by CA's|orig currency|CDI Acumulado":"CDI", "Close|adj by CA's|orig currency|IBXX":"IBXX", "Close|adj by CA's|orig currency|SMLL":"SMLL"}, axis = 1)
bench = bench[all_columns].replace('-', np.NaN)
ibov = bench.dropna()
for c in all_columns[1:]:
    ibov[c] = pd.to_numeric(ibov[c])
ibov['ibov_variation'] = ibov['IBOV'].pct_change()

ibov['Date'] = pd.to_datetime(ibov['Date'])

di=pd.read_excel('taxas_b3.xlsx', usecols=['Data', 'Fator Diário'])

ibov = ibov.set_index('Date').join(di.set_index('Data')).dropna().reset_index().rename({'Fator Diário':'DI'}, axis=1)


# Define os símbolos e as datas de início e fim
indice_US = "^GSPC"  # S&P 500
start_date = ibov['Date'].iloc[0]
end_date = ibov['Date'].iloc[-1]

sp500_data = yf.download(indice_US, start=start_date, end=end_date)
sp500_data_df = pd.DataFrame({'IBOV_US': sp500_data['Adj Close']})

ibov = ibov.set_index('Date').join(sp500_data_df).dropna().reset_index()

taxas_pre = pd.read_csv('taxas_pre.csv')
taxas_pre['date'] = pd.to_datetime(taxas_pre['date'])
taxas_pre.head(25)

# taxas_pre.value_counts('mty')
mty_short_term = 126 #6 meses
mty_long_term = 1260 #5 meses

taxas_pre_short = taxas_pre.loc[taxas_pre['mty']==mty_short_term, ['date', 'taxa']].set_index('date').rename({'taxa':'SHORT_INT'}, axis=1)
taxas_pre_short['SHORT_INT'] = taxas_pre_short['SHORT_INT']*10000
taxas_pre_long = taxas_pre.loc[taxas_pre['mty']==mty_long_term, ['date', 'taxa']].set_index('date').rename({'taxa':'LONG_INT'}, axis=1)
taxas_pre_long['LONG_INT'] = taxas_pre_long['LONG_INT']*10000

ibov = ibov.set_index('Date').join(taxas_pre_short).dropna()
ibov = ibov.join(taxas_pre_long).dropna().reset_index()

taxas_selic = pd.read_csv('taxas_selic.csv', sep=';').rename({'1178 - Taxa de juros - Selic anualizada base 252 - % a.a.':'SELIC'}, axis=1)
taxas_selic.drop(taxas_selic.tail(1).index,inplace=True)
taxas_selic['Data'] = taxas_selic.Data.apply(lambda x: '-'.join(x.split('/')[::-1]))
taxas_selic['Data'] = pd.to_datetime(taxas_selic['Data'])
taxas_selic['SELIC'] = taxas_selic.SELIC.apply(lambda x: x.replace(',', '.'))

taxas_selic = taxas_selic.set_index('Data')

ibov = ibov.set_index('Date').join(taxas_selic).dropna().reset_index()
ibov['SELIC'] = pd.to_numeric(ibov['SELIC'])

lista_selic = list(ibov['SELIC'])
# lista_selic
lista_ciclo = [0 if lista_selic[i]<lista_selic[i-1] else 1 if lista_selic[i]>lista_selic[i-1] else -1 for i in range(1, len(lista_selic))]
last_non_negative = None
new_lista_ciclo = []

for value in lista_ciclo:
    if value != -1:
        last_non_negative = value
    new_lista_ciclo.append(last_non_negative)
new_lista_ciclo = [new_lista_ciclo[0],*new_lista_ciclo]
ibov['SELIC_CYCLE'] = pd.Series(new_lista_ciclo, name='SELIC_CYCLE')
ibov.head(10)


# %%
plt.hist(ibov['LONG_INT'] - ibov['SHORT_INT'], bins=30, edgecolor='black')
plt.show()

plt.plot(ibov['Date'], ibov['SELIC_CYCLE'])
plt.show()

# ibov.value_counts('SELIC_CYCLE')
# %%
def summary_statistics(data):
    """
    Build some basic, pre-defined, statistic results for a series of values.
    """
    if isinstance(data, pd.DataFrame):
        cagrs = {}
        for col in data.columns:
            cagrs[col] = summary_statistics(data[col])
        return pd.DataFrame(cagrs.values(), cagrs.keys())
    assert pd.api.types.is_numeric_dtype(data), f"Expected numeric type. Received `{data.dtype}` dtype."
    res = data.agg(
        [
            np.amin,
            np.amax,
            np.median,
            np.mean,
            np.std,
            lambda x: x.skew(),
            lambda x: x.kurt(),
        ]
    )
    return pd.Series(res.values, ["amin", "amax", "median", "mean", "std", "skew", "kurt"])

# Estratégias
def estrategia_variacao(ser: pd.Series, peso) -> pd.Series:
    starting_pos = 1
#     exp = starting_pos + peso*ser
#     exp[exp<0] = 0
#     return exp
    return starting_pos + peso*ser

def estrategia_juros(ser_ciclo, ser_inclinacao):
    condlist = [
        (ser_ciclo == 0) & (ser_inclinacao >= 67),
        (ser_ciclo == 0) & (ser_inclinacao < 67),
        (ser_ciclo == 1) & (ser_inclinacao >= 183),
        (ser_ciclo == 1) & (ser_inclinacao < 183)
    ]
    # return starting_pos + peso*ser
    choicelist = [0, 1, -1, 0]
    return np.select(condlist, choicelist)

# Sharp
def premio_risco(ser):
    return ser.mean()/ser.std()

# def premio_risco_diversificado(ser, variacao_exp):
#     coef_ret = variacao_exp ** 1.3
#     coef_vol = variacao_exp ** 1.25
#     return (ser.mean()*coef_ret)/(ser.std()*coef_vol)

# %%
window = 2 #dias, 6 meses
null_n_strategy = 0
shift = 2


def gera_graficos(df, init_date = None, end_date = None):
    df = df.loc[:,['Date', 'SHORT_INT', 'LONG_INT', 'SELIC_CYCLE', 'ibov_variation', 'DI']]
    if init_date is not None:
        init_date = pd.to_datetime(init_date, format='%d-%m-%Y')
        df = df.loc[df['Date']>=init_date]
        
    if end_date is not None:
        end_date = pd.to_datetime(end_date, format='%d-%m-%Y')
        df = df.loc[df['Date']<=end_date]
    
    dic_day = pesquisa_multipeso_multidia(df)

    df_resumo = pd.DataFrame.from_dict(dic_day, orient='index', columns = ['Exposição', 'Sharp', 'Peso', 'Atribuição_Estratégia'])
    # figure, axis = plt.subplots(1, 1)
    # figure.set_figwidth(10)
    # figure.set_figheight(20)

    # axis[0].plot(df_resumo.index, df_resumo['Sharp'])
    # axis[0].set_title("Sharp no Tempo")

#     print('Peso')
#     print(df_resumo['Peso'])
    plt.plot(df_resumo.index, df_resumo['Peso'])
    plt.title("Peso no Tempo", fontsize = 20)
#     print('Exposição')
#     print(df_resumo['Exposição'])
    # axis[1].plot(var_variacao[0]['Date'], var_variacao[1], color='blue')
    # axis[1].set_ylabel("Sinal",color="blue",fontsize=14)
    # axis2=axis[1].twinx()
    # axis2.plot(df_resumo.index, df_resumo['Exposição'], color='red')
    # axis2.set_ylabel("Exposição",color="red",fontsize=14)
    # axis2.set_title("Exposição e Sinal no Tempo", fontsize = 20)
    
    plt.show()
    
    premio_risco_variavel = premio_risco(df_resumo[f'Atribuição_Estratégia'])
    
    compare_df = ibov[['IBOV','Date']].set_index('Date').join(np.cumprod(1+df_resumo['Atribuição_Estratégia'])).dropna()

    compare_df['IBOV'] = pd.to_numeric(compare_df['IBOV'])
    compare_df['IBOV'] /= compare_df['IBOV'].iloc[0]
    compare_df[f'Atribuição_Estratégia'] /= compare_df[f'Atribuição_Estratégia'].iloc[0]
#     compare_df['pnl_70%'] /= compare_df['pnl_70%'].iloc[0]
    return premio_risco_variavel, compare_df


#pesquisa desde o começo até o day_referencce
def pesquisa_multipeso_multidia(df):
    dic_day = {}
    i = 0
    for day in df['Date'][window+null_n_strategy+shift:]:
        print(f'Processando dia {day}')
        all_variable_series = df.loc[df['Date']<=day].tail(window+null_n_strategy+shift)
        variable_series = all_variable_series.iloc[:-1]
        # print(variable_series.value_counts('SELIC_CYCLE'))
#         print(variable_series)
        
        # pesos = [-3.0,-2.8,-2.6,-2.4,-2.2,-2,-1.8,-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3]
        # pesos = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        pesos = [1]

        melhor_premio = float('-inf')
        melhor_exposition = 0
        melhor_peso = 0
        melhor_exposition_df = 0
        for peso in pesos:
            print(peso)
            exposition_d_1, premio_risco_variavel = calcula_premio_risco_dia(variable_series, estrategia_juros, peso)
            print(premio_risco_variavel)
            if premio_risco_variavel > melhor_premio:
                melhor_exposition = exposition_d_1
                melhor_premio = premio_risco_variavel
                melhor_peso = peso

        ibov_variation_reference_day = all_variable_series.iloc[-1]['ibov_variation']

        dic_day[day] = [melhor_exposition, melhor_premio, melhor_peso, melhor_exposition*ibov_variation_reference_day]
        # clear_output(wait=True)
        
# #         if i == 800:
        if i == 2:

            break
        i+=1
    return dic_day
     
             
def calcula_premio_risco_dia(variable_series, estrategia, peso):
    this_exposition = variable_series
#     print(this_exposition)
    this_exposition[f'exposicao_juros_inclinacao'] = this_exposition['LONG_INT'] - this_exposition['SHORT_INT'] 

    this_exposition[f'exposicao_juros'] = estrategia(this_exposition[f'SELIC_CYCLE'],this_exposition[f'exposicao_juros_inclinacao'])

    this_exposition[f'exposicao_juros_shift'] = this_exposition[f'exposicao_juros'].shift(2)
    
    this_exposition[f'exposicao_juros_shift_difference'] = this_exposition[f'exposicao_juros_shift'] - 1
    
    this_exposition[f'pnl_juros'] = this_exposition['ibov_variation'] * this_exposition[f'exposicao_juros_shift']
    
    this_exposition.loc[this_exposition[f'exposicao_juros_shift_difference'] > 0, f'pnl_juros'] -= this_exposition['DI'] * (this_exposition[f'exposicao_juros_shift']-1)
    this_exposition.loc[this_exposition[f'exposicao_juros_shift_difference'] < 0, f'pnl_juros'] += this_exposition['DI'] * (1-this_exposition[f'exposicao_juros_shift'])
    
    # print(pd.DataFrame({
    #     'Date':this_exposition['Date'],
    #     'Ciclo':this_exposition['SELIC_CYCLE'],
    #     'Inclinacao':this_exposition[f'exposicao_juros_inclinacao'],
    #     'Exp_pre_shift':this_exposition[f'exposicao_juros'],
    #     'Exp_pos_shift':this_exposition[f'exposicao_juros_shift'],
    #     'IBOV_Var':this_exposition['ibov_variation'],
    #     'P&L_pos_DI':this_exposition[f'pnl_juros']
    # }))
    
    premio_risco_variavel = premio_risco(this_exposition[f'pnl_juros'])

    return this_exposition[f'exposicao_juros_shift'].iloc[-1], premio_risco_variavel


# %%
start_time = time.time()

premio_risco_variavel, compare_df = gera_graficos(ibov, '01-01-2008', '01-10-2010')
print(f'{(time.time() - start_time)/60} minutos')
px.line(compare_df, log_y=True, title = "Performance Variação IBOV_US x IBOV {:.4f}".format(premio_risco_variavel))
# %%
