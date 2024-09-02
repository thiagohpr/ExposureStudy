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
import plotly.io as pio

# Taxa de juros e S&P
# Outra variável comparativa de carteiras - skillness por ex,
#   blend de variáveis, retorno absoluto, minimizar a maior queda

# %%
all_columns = ['Date','IBOV', 'DOLBRL', 'CDI', 'IBXX', 'SMLL']

bench = pd.read_csv('20230224_dao_benchmarks.csv')
bench = bench.rename({"Close|adj by CA's|orig currency|IBOV":"IBOV",
        "Close|adj by CA's|orig currency|DOLOF":"DOLBRL",
        "Close|adj by CA's|orig currency|CDI Acumulado":"CDI",
        "Close|adj by CA's|orig currency|IBXX":"IBXX",
        "Close|adj by CA's|orig currency|SMLL":"SMLL"}, axis = 1)

bench = bench[all_columns].replace('-', np.NaN)
ibov = bench.dropna()
for c in all_columns[1:]:
    ibov[c] = pd.to_numeric(ibov[c])
ibov['ibov_variation'] = ibov['IBOV'].pct_change()

ibov['Date'] = pd.to_datetime(ibov['Date'])

di=pd.read_excel('taxas_b3.xlsx', usecols=['Data', 'Fator Diário'])

ibov = ibov.set_index('Date')
ibov = ibov.join(di.set_index('Data'))
ibov = ibov.dropna().reset_index().rename({'Fator Diário':'DI'}, axis=1)

# Define os símbolos e as datas de início e fim
indice_US = "^GSPC"  # S&P 500
start_date = ibov['Date'].iloc[0]
end_date = ibov['Date'].iloc[-1]

sp500_data = yf.download(indice_US, start=start_date, end=end_date)
sp500_data_df = pd.DataFrame({'IBOV_US': sp500_data['Adj Close']})

US_treasure = yf.download('ZB=F', start=start_date, end=end_date)
US_treasure_df = pd.DataFrame({'US_Treas': US_treasure['Adj Close']})
cmdt = yf.download('GD=F', start=start_date, end=end_date)
cmdt_df = pd.DataFrame({'Commodt': cmdt['Adj Close']})
emerg = yf.download('VWO', start=start_date, end=end_date)
emerg_df = pd.DataFrame({'Emerg': emerg['Adj Close']})
soybean = yf.download('ZS=F', start=start_date, end=end_date)
soybean_df = pd.DataFrame({'soybean': soybean['Adj Close']})

ibov = ibov.set_index('Date').join(sp500_data_df).dropna().reset_index()
ibov = ibov.set_index('Date').join(US_treasure_df).dropna().reset_index()
ibov = ibov.set_index('Date').join(cmdt_df).dropna().reset_index()
ibov = ibov.set_index('Date').join(emerg_df).dropna().reset_index()
ibov = ibov.set_index('Date').join(soybean_df).dropna().reset_index()



taxas_pre = pd.read_csv('taxas_pre.csv')
taxas_pre['date'] = pd.to_datetime(taxas_pre['date'])
taxas_pre.head(25)

# taxas_pre.value_counts('mty')
mty_short_term = 126 #6 meses
mty_long_term = 1260 #5 anos

taxas_pre_short = taxas_pre.loc[taxas_pre['mty']==mty_short_term, ['date', 'taxa']]
taxas_pre_short = taxas_pre_short.set_index('date').rename({'taxa':'SHORT_INT'}, axis=1)
taxas_pre_short['SHORT_INT'] = taxas_pre_short['SHORT_INT']*10000

taxas_pre_long = taxas_pre.loc[taxas_pre['mty']==mty_long_term, ['date', 'taxa']]
taxas_pre_long = taxas_pre_long.set_index('date').rename({'taxa':'LONG_INT'}, axis=1)
taxas_pre_long['LONG_INT'] = taxas_pre_long['LONG_INT']*10000

ibov = ibov.set_index('Date').join(taxas_pre_short).dropna()
ibov = ibov.join(taxas_pre_long).dropna().reset_index()

taxas_selic_name = {'1178 - Taxa de juros - Selic anualizada base 252 - % a.a.':'SELIC'}
taxas_selic = pd.read_csv('taxas_selic.csv', sep=';')
taxas_selic = taxas_selic.rename(taxas_selic_name, axis=1)
taxas_selic.drop(taxas_selic.tail(1).index,inplace=True)
taxas_selic['Data'] = taxas_selic.Data.apply(lambda x: '-'.join(x.split('/')[::-1]))
taxas_selic['Data'] = pd.to_datetime(taxas_selic['Data'])
taxas_selic['SELIC'] = taxas_selic.SELIC.apply(lambda x: x.replace(',', '.'))

taxas_selic = taxas_selic.set_index('Data')

ibov = ibov.set_index('Date').join(taxas_selic).dropna().reset_index()
ibov['SELIC'] = pd.to_numeric(ibov['SELIC'])

lista_selic = list(ibov['SELIC'])
# lista_selic
lista_ciclo = [0 if lista_selic[i]<lista_selic[i-1]
                else 1 if lista_selic[i]>lista_selic[i-1]
                else -1 for i in range(1, len(lista_selic))]

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
# Estratégias
def estrategia_variacao(ser: pd.Series, peso) -> pd.Series:
    starting_pos = 1
    return starting_pos + peso*ser

def estrategia_juros(ser_ciclo, ser_inclinacao, peso):
    condlist = [
        (ser_ciclo == 0) & (ser_inclinacao >= 67),
        (ser_ciclo == 0) & (ser_inclinacao < 67),
        (ser_ciclo == 1) & (ser_inclinacao >= 183),
        (ser_ciclo == 1) & (ser_inclinacao < 183)
    ]
    # return starting_pos + peso*ser
    choicelist = [0, peso*1, peso*-1, 0]
    return np.select(condlist, choicelist)

def estrategia_variacao2(ser1, ser2, peso, ultima_exposicao):
    starting_pos = 1
    if ultima_exposicao>0:
        return starting_pos + peso*ser1
    else:
        return starting_pos + peso*ser2

# Sharp
def premio_risco(ser):
    return ser.mean()/ser.std()
def premio(ser):
    return ser.mean()

# def premio_risco_diversificado(ser, variacao_exp):
#     coef_ret = variacao_exp ** 1.3
#     coef_vol = variacao_exp ** 1.25
#     return (ser.mean()*coef_ret)/(ser.std()*coef_vol)

# %%
window = 180 #dias, 6 meses
null_n_strategy = 1
shift = 2

def custom_function(series):
    return series.mean()  # Placeholder example: Calculating mean of the series

def gera_linha_resumo(strategy ,df, df_resumo):
    # linha_resumo
    # retorno acumulado, meses comprados, meses vendidos, sharpe anual médio(média de todos os sharpes calculados anualmente)
    df_resumo = df_resumo.reset_index()
    df_resumo = df_resumo.rename({'index':'Date'}, axis = 1)

    # print(df_resumo.head(10))
    retorno_acumulado = np.cumprod(1+df_resumo['Atribuição_Estratégia'])
    retorno_acumulado /= retorno_acumulado[0]
    retorno_acumulado = retorno_acumulado.iloc[-1]

    df_resumo['year'] = df_resumo.Date.dt.year
    df_resumo['month'] = df_resumo.Date.dt.month


    monthly_sum = df_resumo.groupby(['year', 'month'])['Exposição'].sum()
    months_negative = (monthly_sum <= 0).sum()
    months_positive = (monthly_sum > 0).sum()

    mean_yearly_sharpe = df_resumo.groupby('year')['Atribuição_Estratégia'].apply(premio_risco).mean()
    # print(f'Retorno Acumulado: {retorno_acumulado}')
    # print(f'Meses Exp<=0: {months_negative}')
    # print(f'Meses Exp>0: {months_positive}')
    # print(f'Shape médio anual: {mean_yearly_sharpe}')
    this_line = pd.Series({'Estratégia':strategy,'Retorno Acumulado':retorno_acumulado, 'Meses Exp<=0':int(months_negative), 'Meses Exp>0':int(months_positive), 'Sharpe Médio Anual':mean_yearly_sharpe})
    df = df.append(this_line, ignore_index=True)
    return df


def gera_grafico_performance(time_df, performance_df, exposicao_df, ibov_df, variable, date_i, save_fig):
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(8)

    legend_lines = []  # List to store plotted lines

    for i in range(len(time_df) - 1):
        x = [time_df[i], time_df[i + 1]]
        y1 = [performance_df[i], performance_df[i + 1]]
        y2 = [ibov_df[i], ibov_df[i + 1]]
        color = 'r' if exposicao_df[i] < 0 else 'b' if exposicao_df[i] < 1 else 'g'  # Set color based on the additional variable
        label = 'Exp<0' if exposicao_df[i] < 0 else 'Exp>0 and Exp<1' if exposicao_df[i] < 1 else 'Exp>=1'
        line, = ax.plot(x, y1,label=label, color=color)
        ax.plot(x, y2, color = 'orange')

        if label not in [line.get_label() for line in legend_lines]:
            legend_lines.append(line)

    plt.xlabel('Time')
    plt.ylabel('Performance')
    plt.title("Performance Variação {} x IBOV {:.4f}".format(variable, premio_risco_variavel))
    plt.legend(handles = legend_lines)
    if save_fig:
        plt.savefig(f'resultados_final/{variable}_{date_i}.jpg')

    plt.show()

def gera_graficos(variable_strategy, df, init_date = None, end_date = None):
    df = df.loc[:,['Date', variable_strategy, 'ibov_variation', 'DI']]
    if init_date is not None:
        init_date = pd.to_datetime(init_date, format='%d-%m-%Y')
        df = df.loc[df['Date']>=init_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date, format='%d-%m-%Y')
        df = df.loc[df['Date']<=end_date]

    dic_day, var_variacao = pesquisa_multipeso_multidia(variable_strategy, df)

    df_resumo = pd.DataFrame.from_dict(dic_day, orient='index', columns = ['Exposição', 'Sharp', 'Peso', 'Atribuição_Estratégia'])
#     figure, axis = plt.subplots(2, 1)
#     figure.set_figwidth(10)
#     figure.set_figheight(20)

#     # axis[0].plot(df_resumo.index, df_resumo['Sharp'])
#     # axis[0].set_title("Sharp no Tempo")

# #     print('Peso')
# #     print(df_resumo['Peso'])
#     axis[0].plot(df_resumo.index, df_resumo['Peso'])
#     axis[0].set_title("Peso no Tempo", fontsize = 20)
# #     print('Exposição')
# #     print(df_resumo['Exposição'])
#     axis[1].plot(var_variacao[0]['Date'], var_variacao[1], color='blue')
#     axis[1].set_ylabel("Sinal",color="blue",fontsize=14)
#     axis2=axis[1].twinx()
#     axis2.plot(df_resumo.index, df_resumo['Exposição'], color='red')
#     axis2.set_ylabel("Exposição",color="red",fontsize=14)
#     axis2.set_title("Exposição e Sinal no Tempo", fontsize = 20)
#     plt.savefig(f'resultados_interno2/{variable_strategy}_aux.jpg')
#     plt.show()

    premio_risco_variavel = premio_risco(df_resumo[f'Atribuição_Estratégia'])

    compare_df = ibov[['IBOV','Date']].set_index('Date').join(np.cumprod(1+df_resumo['Atribuição_Estratégia'])).dropna()

    compare_df['IBOV'] = pd.to_numeric(compare_df['IBOV'])
    compare_df['IBOV'] /= compare_df['IBOV'].iloc[0]
    compare_df[f'Atribuição_Estratégia'] /= compare_df[f'Atribuição_Estratégia'].iloc[0]
#     compare_df['pnl_70%'] /= compare_df['pnl_70%'].iloc[0]
    # return premio_risco_variavel, compare_df
    return premio_risco_variavel, compare_df, df_resumo




#pesquisa desde o começo até o day_referencce
def pesquisa_multipeso_multidia(variable_strategy, df):
    dic_day = {}
    i = 0
    for day in df['Date'][window+null_n_strategy+shift:]:
        print(f'Processando dia {day}')
        all_variable_series = df.loc[df['Date']<=day].tail(window+null_n_strategy+shift)
        variable_series = all_variable_series.iloc[:-1]
#         print(variable_series)
        
        variable_series[f'{variable_strategy}_variation'] = np.cumprod(1+variable_series[variable_strategy].pct_change())
        # variable_series[f'{variable_strategy}_variation'] = variable_series[variable_strategy].pct_change()
 
        #normaliza sinal de 0 a 1
        variable_series[f'{variable_strategy}_variation'] = variable_series[f'{variable_strategy}_variation']/variable_series[f'{variable_strategy}_variation'].max()
        
        pesos = [-4,-3.0,-2.8,-2.6,-2.4,-2.2,-2,-1.8,-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,4]
        # pesos = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

        melhor_premio = -99999999999
        melhor_exposition = 0
        melhor_peso = 0
        melhor_exposition_df = 0
        for peso in pesos:
            exposition_d_1, premio_risco_variavel = calcula_premio_risco_dia(variable_series, variable_strategy, estrategia_variacao, peso)

            if premio_risco_variavel > melhor_premio:
                melhor_exposition = exposition_d_1
                melhor_premio = premio_risco_variavel
                melhor_peso = peso

        ibov_variation_reference_day = all_variable_series.iloc[-1]['ibov_variation']

        dic_day[day] = [melhor_exposition, melhor_premio, melhor_peso, melhor_exposition*ibov_variation_reference_day]
        clear_output(wait=True)
        
# #         if i == 800:
        # if i == 0:

        #     break
        # i+=1
    return dic_day, [df[df['Date'].isin(list(dic_day.keys()))], df[df['Date'].isin(list(dic_day.keys()))][variable_strategy].pct_change()]
     
             
def calcula_premio_risco_dia(variable_series, variable_strategy, estrategia, peso):
    this_exposition = variable_series
#     print(this_exposition)

    this_exposition[f'exposicao_variacao_{variable_strategy}'] = estrategia(this_exposition[f'{variable_strategy}_variation'], peso)

    this_exposition[f'exposicao_variacao_{variable_strategy}_shift'] = this_exposition[f'exposicao_variacao_{variable_strategy}'].shift(2)
    
    this_exposition[f'exposicao_variacao_{variable_strategy}_shift_difference'] = this_exposition[f'exposicao_variacao_{variable_strategy}_shift'] - 1
    
    this_exposition[f'pnl_variacao_{variable_strategy}'] = this_exposition['ibov_variation'] * this_exposition[f'exposicao_variacao_{variable_strategy}_shift']
    
    this_exposition.loc[this_exposition[f'exposicao_variacao_{variable_strategy}_shift_difference'] > 0, f'pnl_variacao_{variable_strategy}'] -= this_exposition['DI'] * (this_exposition[f'exposicao_variacao_{variable_strategy}_shift']-1)
    this_exposition.loc[this_exposition[f'exposicao_variacao_{variable_strategy}_shift_difference'] < 0, f'pnl_variacao_{variable_strategy}'] += this_exposition['DI'] * (1-this_exposition[f'exposicao_variacao_{variable_strategy}_shift'])
    
    # print(pd.DataFrame({
    #     'Variavel':this_exposition[variable_strategy],
    #     'Sinal':this_exposition[f'{variable_strategy}_variation'],
    #     'Exp_pre_shift':this_exposition[f'exposicao_variacao_{variable_strategy}'],
    #     'Exp_pos_shift':this_exposition[f'exposicao_variacao_{variable_strategy}_shift'],
    #     'IBOV_Var':this_exposition['ibov_variation'],
    #     'P&L_pos_DI':this_exposition[f'pnl_variacao_{variable_strategy}']
    # }))
    
    premio_risco_variavel = premio_risco(this_exposition[f'pnl_variacao_{variable_strategy}'])

    return this_exposition[f'exposicao_variacao_{variable_strategy}_shift'].iloc[-1], premio_risco_variavel


# %%

def gera_graficos2(variable_strategy1, variable_strategy2, df, init_date = None, end_date = None):
    df = df.loc[:,['Date', variable_strategy1, variable_strategy2, 'ibov_variation', 'DI']]
    if init_date is not None:
        init_date = pd.to_datetime(init_date, format='%d-%m-%Y')
        df = df.loc[df['Date']>=init_date]
        
    if end_date is not None:
        end_date = pd.to_datetime(end_date, format='%d-%m-%Y')
        df = df.loc[df['Date']<=end_date]
    
    dic_day = pesquisa_multipeso_multidia2(variable_strategy1, variable_strategy2, df)

    df_resumo = pd.DataFrame.from_dict(dic_day, orient='index', columns = ['Exposição', 'Sharp', 'Peso', 'Atribuição_Estratégia'])
    
    premio_risco_variavel = premio_risco(df_resumo[f'Atribuição_Estratégia'])
    
    compare_df = ibov[['IBOV','Date']].set_index('Date').join(np.cumprod(1+df_resumo['Atribuição_Estratégia'])).dropna()

    compare_df['IBOV'] = pd.to_numeric(compare_df['IBOV'])
    compare_df['IBOV'] /= compare_df['IBOV'].iloc[0]
    compare_df[f'Atribuição_Estratégia'] /= compare_df[f'Atribuição_Estratégia'].iloc[0]
#     compare_df['pnl_70%'] /= compare_df['pnl_70%'].iloc[0]
    # return premio_risco_variavel, compare_df
    return premio_risco_variavel, compare_df, df_resumo


def pesquisa_multipeso_multidia2(variable_strategy1, variable_strategy2, df):
    dic_day = {}
    i = 0
    ultima_exposicao = 1
    for day in df['Date'][window+null_n_strategy+shift:]:
        print(f'Processando dia {day}')
        all_variable_series = df.loc[df['Date']<=day].tail(window+null_n_strategy+shift)
        variable_series = all_variable_series.iloc[:-1]
#         print(variable_series)
        
        variable_series[f'{variable_strategy1}_variation'] = np.cumprod(1+variable_series[variable_strategy1].pct_change())
        variable_series[f'{variable_strategy2}_variation'] = np.cumprod(1+variable_series[variable_strategy2].pct_change())

        # variable_series[f'{variable_strategy}_variation'] = variable_series[variable_strategy].pct_change()
 
        #normaliza sinal de 0 a 1
        variable_series[f'{variable_strategy1}_variation'] = variable_series[f'{variable_strategy1}_variation']/variable_series[f'{variable_strategy1}_variation'].max()
        variable_series[f'{variable_strategy2}_variation'] = variable_series[f'{variable_strategy2}_variation']/variable_series[f'{variable_strategy2}_variation'].max()

        
        pesos = [-4,-3.0,-2.8,-2.6,-2.4,-2.2,-2,-1.8,-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,4]
        # pesos = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

        melhor_premio = -99999999999
        melhor_exposition = 0
        melhor_peso = 0
        melhor_exposition_df = 0
        for peso in pesos:
            exposition_d_1, premio_risco_variavel = calcula_premio_risco_dia2(variable_series, variable_strategy1, variable_strategy2, estrategia_variacao2, peso, ultima_exposicao)

            if premio_risco_variavel > melhor_premio:
                melhor_exposition = exposition_d_1
                melhor_premio = premio_risco_variavel
                melhor_peso = peso

        ibov_variation_reference_day = all_variable_series.iloc[-1]['ibov_variation']

        ultima_exposicao = melhor_exposition

        dic_day[day] = [melhor_exposition, melhor_premio, melhor_peso, melhor_exposition*ibov_variation_reference_day]
        clear_output(wait=True)
        
# #         if i == 800:
        # if i == 0:

        #     break
        # i+=1
    return dic_day
     
             
def calcula_premio_risco_dia2(variable_series, variable_strategy1, variable_strategy2, estrategia, peso, ultima_exposicao):
    this_exposition = variable_series
#     print(this_exposition)

    this_exposition[f'exposicao'] = estrategia(this_exposition[f'{variable_strategy1}_variation'], this_exposition[f'{variable_strategy2}_variation'], peso, ultima_exposicao)

    this_exposition[f'exposicao_shift'] = this_exposition[f'exposicao'].shift(2)
    
    this_exposition[f'exposicao_shift_difference'] = this_exposition[f'exposicao_shift'] - 1
    
    this_exposition[f'pnl'] = this_exposition['ibov_variation'] * this_exposition[f'exposicao_shift']
    
    this_exposition.loc[this_exposition[f'exposicao_shift_difference'] > 0, f'pnl'] -= this_exposition['DI'] * (this_exposition[f'exposicao_shift']-1)
    this_exposition.loc[this_exposition[f'exposicao_shift_difference'] < 0, f'pnl'] += this_exposition['DI'] * (1-this_exposition[f'exposicao_shift'])
    
    # print(pd.DataFrame({
    #     'Variavel':this_exposition[variable_strategy],
    #     'Sinal':this_exposition[f'{variable_strategy}_variation'],
    #     'Exp_pre_shift':this_exposition[f'exposicao_variacao_{variable_strategy}'],
    #     'Exp_pos_shift':this_exposition[f'exposicao_variacao_{variable_strategy}_shift'],
    #     'IBOV_Var':this_exposition['ibov_variation'],
    #     'P&L_pos_DI':this_exposition[f'pnl_variacao_{variable_strategy}']
    # }))
    
    premio_risco_variavel = premio_risco(this_exposition[f'pnl'])

    return this_exposition[f'exposicao_shift'].iloc[-1], premio_risco_variavel


# %%
print(list(ibov.columns))


# %%
start_time = time.time()
variable = 'Emerg'

premio_risco_variavel, compare_df, df_geral = gera_graficos(variable, ibov)
print(f'{(time.time() - start_time)/60} minutos')

time_df = compare_df.reset_index()['Date']
performance_df = compare_df['Atribuição_Estratégia']
exposicao_df = df_geral['Exposição']
ibov_df = compare_df['IBOV']
gera_grafico_performance(time_df, performance_df, exposicao_df, ibov_df, variable)


# %%
start_time = time.time()
# variables = ['DOLBRL','IBOV_US','US_Treas','Commodt','LONG_INT','Emerg']
variables = ['soybean']



dates = [['01-01-2007','28-10-2010'],
         ['01-01-2012','28-10-2015'],
         ['01-01-2019','28-10-2022']
]

for variable in variables:
    time_df = []
    performance_df = []
    exposicao_df = []
    ibov_df = []
    for date in dates:
        premio_risco_variavel, compare_df, df_geral = gera_graficos(variable, ibov, date[0],date[1])
        print(f'{(time.time() - start_time)/60} minutos')

        time_df.append(compare_df.reset_index()['Date'])
        performance_df.append(compare_df['Atribuição_Estratégia'])
        exposicao_df.append(df_geral['Exposição'])
        ibov_df.append(compare_df['IBOV'])

    for i in range(len(dates)):
        gera_grafico_performance(time_df[i], performance_df[i], exposicao_df[i], ibov_df[i], variable, i)

# %%
start_time = time.time()
variable1 = 'Emerg'
variable2 = 'DOLBRL'

premio_risco_variavel, compare_df, df_geral = gera_graficos2(variable1, variable2, ibov, '01-01-2007','28-10-2010')
print(f'{(time.time() - start_time)/60} minutos')

time_df = compare_df.reset_index()['Date']
performance_df = compare_df['Atribuição_Estratégia']
exposicao_df = df_geral['Exposição']
ibov_df = compare_df['IBOV']
gera_grafico_performance(time_df, performance_df, exposicao_df, ibov_df, variable1+'-'+variable2, 3, False)
# %%
start_time = time.time()
variable1 = 'soybean'
variable2 = 'LONG_INT'

dates = [['01-01-2007','28-10-2010'],
         ['01-01-2012','28-10-2015'],
         ['01-01-2019','28-10-2022']
]

time_df = []
performance_df = []
exposicao_df = []
ibov_df = []

for date in dates:
    premio_risco_variavel, compare_df, df_geral = gera_graficos2(variable1, variable2, ibov, date[0],date[1])
    print(f'{(time.time() - start_time)/60} minutos')

    time_df.append(compare_df.reset_index()['Date'])
    performance_df.append(compare_df['Atribuição_Estratégia'])
    exposicao_df.append(df_geral['Exposição'])
    ibov_df.append(compare_df['IBOV'])
for i in range(len(dates)):
    gera_grafico_performance(time_df[i], performance_df[i], exposicao_df[i], ibov_df[i], variable1+'-'+variable2, i)


# %%

df_estrategias = pd.DataFrame()

start_time = time.time()
variables = ['DOLBRL','IBOV_US','US_Treas','Commodt','LONG_INT','Emerg', 'soybean']

df_gerais = []

for variable in variables:
    premio_risco_variavel, compare_df, df_geral = gera_graficos(variable, ibov)
    df_estrategias = gera_linha_resumo(variable, df_estrategias, df_geral)
    df_gerais.append(df_geral)

print(f'{(time.time() - start_time)/60} minutos')
print(df_estrategias)
df_estrategias.to_csv('resultados_final/medidas_resumo.csv')


# %%
compare_df.tail(10)
# %%
