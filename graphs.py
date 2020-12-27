import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


def graphLabels(xList, yList, legends, title, xlabel, ylabel, name, show=True, semilogy=False):
    if semilogy:
        for x, y in zip(xList, yList):
            plt.semilogy(x, y)
    else:
        for x, y in zip(xList, yList):
            plt.plot(x, y)
    if len(legends) != 0:
        plt.legend(legends)
    if title != '':
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.autoscale(tight=True)
    plt.grid()
    plt.savefig("images/" + name)
    if show:
        plt.show()


data = json.load(open('data_weighted.json'))
config = json.load(open('config.json'))
prices = pd.read_csv('Data/electricity_prices.csv')  # Prices dataframe
PV_production = pd.read_csv('Data/PVproduction_PanelSize1kWp.csv')  # Output PV power dataframe

Tmax_vector = np.linspace(config['without_PV']['Tmax']['start'], config['without_PV']['Tmax']['finish'], config['with_PV']['Tmax']['samples'])
f_vector = np.linspace(config['without_PV']['f']['start'], config['without_PV']['f']['finish'], config['without_PV']['f']['samples'])

# %% without Solar Panels
for key in config.keys():
    for season in ['winter', 'summer']:
        # Tmax
        graphLabels([np.array(data[key]['Tmax'][f'Time_{Tmax}_{season}']) for Tmax in Tmax_vector], [data[key]['Tmax'][f'LossProb_{Tmax}_{season}'] for Tmax in Tmax_vector], [f'Tmax: '+f'{int(Tmax)} min' for Tmax in Tmax_vector], f'Loss Probability with f=0.5 in {season}', 'Time of the day', 'Loss Probability', f'lossProb_Tmax_{key}_{season}.pdf')

        graphLabels([np.array(data[key]['Tmax'][f'Time_{Tmax}_{season}']) for Tmax in Tmax_vector], [data[key]['Tmax'][f'Cost_{Tmax}_{season}'] for Tmax in Tmax_vector], [f'Tmax: '+f'{int(Tmax)} min' for Tmax in Tmax_vector], f'Cost with f=0.5 in {season}', 'Time of the day', 'Cumulative expenditure (euro/MWh)', f'cost_Tmax_{key}_{season}.pdf')

        graphLabels([np.array(data[key]['Tmax'][f'Time_{Tmax}_{season}']) for Tmax in Tmax_vector], [data[key]['Tmax'][f'Working_{Tmax}_{season}'] for Tmax in Tmax_vector], [f'Tmax: '+f'{int(Tmax)} min' for Tmax in Tmax_vector], f'Number of working charging stations in {season}', 'Time of the day', 'Number of working charging stations', f'working_Tmax_{key}_{season}.pdf')

        # f
        graphLabels([np.array(data[key]['f'][f'Time_{f}_{season}']) for f in f_vector], [data[key]['f'][f'LossProb_{f}_{season}'] for f in f_vector], [f'Fraction: '+f'{int(f*100)}%' for f in f_vector], f'Loss Probability with Tmax=60 in {season}', 'Time of the day', 'Loss Probability', f'lossProb_f_{key}_{season}.pdf')

        graphLabels([np.array(data[key]['f'][f'Time_{f}_{season}']) for f in f_vector], [data[key]['f'][f'Cost_{f}_{season}'] for f in f_vector], [f'Fraction: '+f'{int(f*100)}%' for f in f_vector], f'Cost with Tmax=60 in {season}', 'Time of the day', 'Cumulative expenditure (euro/MWh)', f'cost_f_{key}_{season}.pdf')

        graphLabels([np.array(data[key]['f'][f'Time_{f}_{season}']) for f in f_vector], [data[key]['f'][f'Working_{f}_{season}'] for f in f_vector], [f'Fraction: '+f'{int(f*100)}%' for f in f_vector], f'Number of working charging stations with Tmax=60 in {season}', 'Time of the day', 'Number of working charging stations', f'working_f_{key}_{season}.pdf')

# Costs graph
high_costs = [[(np.max(prices[prices['Season'] == season]['Cost']) - np.std(prices[prices['Season'] == season]['Cost']))*0.9]*24 for season in ['WINTER', 'SUMMER']]
season_costs = [prices[prices['Season'] == season]['Cost'] for season in ['WINTER', 'SUMMER']]
costs = [high_costs, season_costs]
graphLabels([prices['Hour'][:24]]*4, [costs[0][0], costs[0][1], costs[1][0], costs[1][1]], ['High Cost Winter', 'High Cost Summer', 'Cost in Winter', 'Cost in Summer'], 'Costs per Season', 'Time of the day', 'Cost (euro/MWh)', 'costs_per_season.pdf')
# Power generated graph
power_generation = [PV_production[(PV_production['Month'] == month) & (PV_production['Day'] == day)]['Output power (W)'] for month, day in [(1,1),(6,26)]]
graphLabels([prices['Hour'][:24]]*2, [power_generation[i].values for i in range(2)], ['January the 1st', 'June the 26th'], 'Power generation', 'Time of the day', 'Output power (W)', 'output_pow_season.pdf')