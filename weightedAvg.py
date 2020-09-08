import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
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
    plt.autoscale(tight=True)
    plt.grid()
    plt.savefig("images/" + name)
    if show:
        plt.show()


data = json.load(open('data_weighted.json'))
config = json.load(open('config.json'))

Tmax_vector = np.linspace(config['without_PV']['Tmax']['start'], config['without_PV']['Tmax']['finish'], config['with_PV']['Tmax']['samples'])
f_vector = np.linspace(config['without_PV']['f']['start'], config['without_PV']['f']['finish'], config['without_PV']['f']['samples'])

# %% without Solar Panels

# Tmax
graphLabels([np.array(data['without_PV']['Tmax'][f'Time_{Tmax}']) for Tmax in Tmax_vector], [data['without_PV']['Tmax'][f'LossProb_{Tmax}'] for Tmax in Tmax_vector], [f'Tmax: '+f'{Tmax}h' for Tmax in Tmax_vector], 'Loss Probability for different Tmax', 'Time of the day', 'Loss Probability', 'lossProb_Tmax_without_PV.pdf')

graphLabels([np.array(data['without_PV']['Tmax'][f'Time_{Tmax}']) for Tmax in Tmax_vector], [data['without_PV']['Tmax'][f'Cost_{Tmax}'] for Tmax in Tmax_vector], [f'Tmax: '+f'{Tmax}h' for Tmax in Tmax_vector], 'Cost for different Tmax', 'Time of the day', 'Cumulative expenditure (euro/MWh)', 'cost_Tmax_without_PV.pdf')

# f
end = 11
graphLabels([np.array(data['without_PV']['f'][f'Time_{f}']) for f in f_vector[:end]], [data['without_PV']['f'][f'LossProb_{f}'] for f in f_vector[:end]], [f'Fraction: '+f'{f*100}' for f in f_vector[:end]], 'Loss Probability for different fraction of non working batteries', 'Time of the day', 'Loss Probability', 'lossProb_f_without_PV.pdf')

graphLabels([np.array(data['without_PV']['f'][f'Time_{f}']) for f in f_vector[:end]], [data['without_PV']['f'][f'Cost_{f}'] for f in f_vector[:end]], [f'Fraction: '+f'{f*100}' for f in f_vector[:end]], 'Cost for different fraction of non working batteries', 'Time of the day', 'Cumulative expenditure (euro/MWh)', 'cost_f_without_PV.pdf')

# Working batteries f
# graphLabels([np.array(data['without_PV']['f'][f'Time_{f}'])/60 for f in f_vector], [data['without_PV']['f'][f'Working_{f}'] for f in f_vector], [f'Fraction: '+f'{f*100}' for f in f_vector], 'Number of working batteries for different fraction of non working batteries', 'Time of the day', 'Number of working charging stations', 'working_f_without_PV.pdf')

# %% with Solar Panels
# f
graphLabels([np.array(data['with_PV']['f'][f'Time_{f}']) for f in f_vector[:end]], [data['with_PV']['f'][f'LossProb_{f}'] for f in f_vector[:end]], [f'Fraction: '+f'{f*100}' for f in f_vector[:end]], 'Loss Probability for different fraction of non working batteries', 'Time of the day', 'Loss Probability', 'lossProb_f_with_PV.pdf')

# Working batteries f
# graphLabels([np.array(data['with_PV']['f'][f'Time_{f}'])/60 for f in f_vector], [data['with_PV']['f'][f'Working_{f}'] for f in f_vector], [f'Fraction: '+f'{f*100}' for f in f_vector], 'Number of working batteries for different fraction of non working batteries', 'Time of the day', 'Number of working charging stations', 'working_f_with_PV.pdf')import json
import numpy as np

data = json.load(open('data.json'))
config = json.load(open('config.json'))

# %%

# Extracting all data

for key in config.keys():
    f_vector = np.linspace(config[key]['f']['start'], config[key]['f']['finish'], config[key]['f']['samples'])
    Tmax_vector = np.linspace(config[key]['Tmax']['start'], config[key]['Tmax']['finish'], config[key]['Tmax']['samples'])
    for Tmax in Tmax_vector:
        all_seedsTime = []
        all_seedsLossProb = []
        all_seedsCost = []
        all_seedsWorking = []
        [all_seedsTime.append(np.array(data[key]['Tmax'][f'Time_{Tmax}_{seed}']) / 60) for seed in range(50)]
        [all_seedsLossProb.append(np.array(data[key]['Tmax'][f'LossProb_{Tmax}_{seed}'])) for seed in range(50)]
        [all_seedsCost.append(np.array(data[key]['Tmax'][f'Cost_{Tmax}_{seed}'])) for seed in range(50)]
        [all_seedsWorking.append(np.array(data[key]['Tmax'][f'Working_{Tmax}_{seed}']) / 60) for seed in range(50)]
        # Tuple Time-LossProb
        config[key]['Tmax'][f'Time_LossProb_{Tmax}'] = sorted(zip(np.concatenate(all_seedsTime).tolist(), np.concatenate(all_seedsLossProb).tolist()))
        config[key]['Tmax'][f'Time_{Tmax}'] = [element for element, _ in config[key]['Tmax'][f'Time_LossProb_{Tmax}']]
        config[key]['Tmax'][f'allLossProb_{Tmax}'] = [element for _, element in config[key]['Tmax'][f'Time_LossProb_{Tmax}']]
        # Tuple Time-Costs
        config[key]['Tmax'][f'Time_Cost_{Tmax}'] = sorted(zip(np.concatenate(all_seedsTime).tolist(), np.concatenate(all_seedsCost).tolist()))
        config[key]['Tmax'][f'allCost_{Tmax}'] = [element for _, element in config[key]['Tmax'][f'Time_Cost_{Tmax}']]
        # Tuple Time-Working
        config[key]['Tmax'][f'Time_Working_{Tmax}'] = sorted(zip(np.concatenate(all_seedsTime).tolist(), np.concatenate(all_seedsWorking).tolist()))
        config[key]['Tmax'][f'allWork_{Tmax}'] = [element for _, element in config[key]['Tmax'][f'Time_Working_{Tmax}']]
    for f in f_vector:
        all_seedsTime = []
        all_seedsLossProb = []
        all_seedsCost = []
        [all_seedsTime.append(np.array(data[key]['f'][f'Time_{f}_{seed}']) / 60) for seed in range(50)]
        [all_seedsLossProb.append(np.array(data[key]['f'][f'LossProb_{f}_{seed}'])) for seed in range(50)]
        [all_seedsCost.append(np.array(data[key]['f'][f'Cost_{f}_{seed}'])) for seed in range(50)]
        # Tuple Time-LossProb
        config[key]['f'][f'Time_LossProb_{f}'] = sorted(zip(np.concatenate(all_seedsTime).tolist(), np.concatenate(all_seedsLossProb).tolist()))
        config[key]['f'][f'Time_{f}'] = [element for element, _  in config[key]['f'][f'Time_LossProb_{f}']]
        config[key]['f'][f'allLossProb_{f}'] = [element for _, element in config[key]['f'][f'Time_LossProb_{f}']]
        # Tuple Time-Costs
        config[key]['f'][f'Time_Cost_{f}'] = sorted(zip(np.concatenate(all_seedsTime).tolist(), np.concatenate(all_seedsCost).tolist()))
        config[key]['f'][f'allCost_{f}'] = [element for _, element in config[key]['f'][f'Time_Cost_{f}']]
    # Exponentially weighted moving average
    beta = 0.999
    for Tmax in Tmax_vector:
        weighted_lossProb = []
        weigthed_cost = []
        loss = 0
        cost = 0
        i = 0
        while i < len(config[key]['Tmax'][f'allLossProb_{Tmax}']):
            loss = beta*loss + (1-beta)*config[key]['Tmax'][f'allLossProb_{Tmax}'][i]
            cost = beta*cost + (1-beta)*config[key]['Tmax'][f'allCost_{Tmax}'][i]
            # Bias correction
            # loss = loss/(1-beta**(i+1))
            # cost = cost/(1-beta**(i+1))
            weighted_lossProb.append(loss)
            weigthed_cost.append(cost)
            i += 1
        config[key]['Tmax'][f'LossProb_{Tmax}'] = weighted_lossProb
        config[key]['Tmax'][f'Cost_{Tmax}'] = weigthed_cost
    for f in f_vector:
        weighted_lossProb = []
        weigthed_cost = []
        loss = 0
        cost = 0
        i = 0
        while i < len(config[key]['f'][f'allLossProb_{f}']):
            loss = beta*loss + (1-beta)*config[key]['f'][f'allLossProb_{f}'][i]
            cost = beta*cost + (1-beta)*config[key]['f'][f'allCost_{f}'][i]
            # Bias correction
            # loss = loss / (1 - beta ** (i+1))
            # cost = cost / (1 - beta ** (i+1))
            weighted_lossProb.append(loss)
            weigthed_cost.append(cost)
            i += 1
        config[key]['f'][f'LossProb_{f}'] = weighted_lossProb
        config[key]['f'][f'Cost_{f}'] = weigthed_cost
    json_data = json.dumps(config, indent=4)
    with open('data_weighted.json', 'w') as outfile:
        outfile.write(json_data)
