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

ibov.head(10)



# %%

init_date = pd.to_datetime('2010-1-1', format='%Y-%m-%d')
end_date = pd.to_datetime('2014-12-31', format='%Y-%m-%d')

ibov = ibov.loc[(ibov['Date']>=init_date)&(ibov['Date']<=end_date)]
ibov.head(10)


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
    return starting_pos + peso*ser

def estrategia_simples(ser: pd.Series) -> pd.Series:
    condlist = [
        ser < 0,
        ser == 0,
        ser > 0,
    ]
    choicelist = [0, 0, 1]
    result = np.select(condlist, choicelist)
    return result

# Sharp
def premio_risco(ser):
    return ser.mean()/ser.std()


# %%
window = 1 #dias
shift = 0
null_n_strategy = 0

#pesquisa desde o começo até o day_referencce
def pesquisa_multipeso_multidia(variable_strategy):
    important_columns = ibov.loc[:,['Date', variable_strategy, 'ibov_variation', 'DI']]
    dic_day = {}
    i = 0
    for day in important_columns['Date'][window+null_n_strategy+shift:]:
        print(f'Processando dia {day}')
        variable_series = important_columns.loc[important_columns['Date']<=day].tail(window+null_n_strategy+shift)

        variable_series[f'{variable_strategy}_variation'] = variable_series['ibov_variation'] #mudar parametro retorno acumulado (não usar valores pertos tipo 31,32,33 mas sim 30,60,90,120 para dar informações diferentes)

        peso = 1

        
        exposition_d_1, premio_risco_variavel = calcula_premio_risco_dia(variable_series, variable_strategy, estrategia_simples, peso)

        ibov_variation_reference_day = variable_series.iloc[-1]['ibov_variation']

        dic_day[day] = [exposition_d_1, premio_risco_variavel, peso, exposition_d_1*ibov_variation_reference_day]
        # clear_output(wait=True)
        
                #Debug

# #         if i == 800:
        if i == 200:

            break
        i+=1
    return dic_day, [important_columns[important_columns['Date'].isin(list(dic_day.keys()))], important_columns[important_columns['Date'].isin(list(dic_day.keys()))][variable_strategy].pct_change()]
             

# Gera comparação com o IBOV
def calcula_premio_risco_dia(variable_series, variable_strategy, estrategia, peso):
    #variable_series = ibov.loc[variable_strategy]
    #valores de D0 até D-1
    
    this_exposition = variable_series

    this_exposition[f'exposicao_variacao_{variable_strategy}'] = estrategia(this_exposition[f'{variable_strategy}_variation'])

    this_exposition[f'exposicao_variacao_{variable_strategy}_shift'] = this_exposition[f'exposicao_variacao_{variable_strategy}'].shift(shift)

    
    this_exposition[f'exposicao_variacao_{variable_strategy}_shift_difference'] = this_exposition[f'exposicao_variacao_{variable_strategy}_shift'] - 1
    
    this_exposition[f'pnl_variacao_{variable_strategy}'] = this_exposition['ibov_variation'] * this_exposition[f'exposicao_variacao_{variable_strategy}_shift']
    
    this_exposition.loc[this_exposition[f'exposicao_variacao_{variable_strategy}_shift_difference'] > 0, f'pnl_variacao_{variable_strategy}'] -= this_exposition['DI'] * (this_exposition[f'exposicao_variacao_{variable_strategy}_shift']-1)
    this_exposition.loc[this_exposition[f'exposicao_variacao_{variable_strategy}_shift_difference'] < 0, f'pnl_variacao_{variable_strategy}'] += this_exposition['DI'] * (1-this_exposition[f'exposicao_variacao_{variable_strategy}_shift'])

    
    print(pd.DataFrame({
        'Exp_pre_shift':this_exposition[f'exposicao_variacao_{variable_strategy}'],
        'Exp_pos_shift':this_exposition[f'exposicao_variacao_{variable_strategy}_shift'],
        'IBOV_Var':this_exposition['ibov_variation'],
        'DI':this_exposition['DI'],
        'P&L':this_exposition[f'pnl_variacao_{variable_strategy}']
    }))

    premio_risco_variavel = premio_risco(this_exposition[f'pnl_variacao_{variable_strategy}'])
    
    
    return this_exposition[f'exposicao_variacao_{variable_strategy}_shift'].iloc[-1], premio_risco_variavel
    


def gera_graficos(variable_strategy):
    dic_day, var_variacao = pesquisa_multipeso_multidia(variable_strategy)

    df_resumo = pd.DataFrame.from_dict(dic_day, orient='index', columns = ['Exposição', 'Sharp', 'Peso', 'Atribuição_Estratégia'])
    figure, axis = plt.subplots(3, 1)
    figure.set_figwidth(10)
    figure.set_figheight(20)


    axis[0].plot(df_resumo.index, df_resumo['Sharp'])
    axis[0].set_title("Sharp no Tempo")
#     print('Peso')
#     print(df_resumo['Peso'])
    axis[1].plot(df_resumo.index, df_resumo['Peso'])
    axis[1].set_title("Peso no Tempo")
#     print('Exposição')
#     print(df_resumo['Exposição'])
    axis[2].plot(var_variacao[0]['Date'], var_variacao[1], color='blue')
    axis[2].set_ylabel("Sinal",color="blue",fontsize=14)
    axis2=axis[2].twinx()
    axis2.plot(df_resumo.index, df_resumo['Exposição'], color='red')
    axis2.set_ylabel("Exposição",color="red",fontsize=14)
    axis2.set_title("Exposição e Sinal no Tempo")
    
    plt.show()
    
    premio_risco_variavel = premio_risco(df_resumo[f'Atribuição_Estratégia'])
    
    compare_df = ibov[['IBOV','Date']].set_index('Date').join(np.cumprod(1+df_resumo['Atribuição_Estratégia'])).dropna()

    compare_df['IBOV'] = pd.to_numeric(compare_df['IBOV'])
    compare_df['IBOV'] /= compare_df['IBOV'].iloc[0]
    compare_df[f'Atribuição_Estratégia'] /= compare_df[f'Atribuição_Estratégia'].iloc[0]
#     compare_df['pnl_70%'] /= compare_df['pnl_70%'].iloc[0]
    return premio_risco_variavel, compare_df

# %%
start_time = time.time()

premio_risco_variavel, compare_df = gera_graficos('IBOV_US')
print(f'{(time.time() - start_time)/60} minutos')
px.line(compare_df, log_y=True, title = "Performance Variação Câmbio x IBOV {:.4f}".format(premio_risco_variavel))
# %%
