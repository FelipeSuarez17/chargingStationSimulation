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


data = json.load(open('data.json'))
config = json.load(open('config.json'))

# %% without Solar Panels
Tmax_vector = np.linspace(config['without_PV']['Tmax']['start'], config['without_PV']['Tmax']['finish'], config['with_PV']['Tmax']['samples'])
f_vector = np.linspace(config['without_PV']['f']['start'], config['without_PV']['f']['finish'], config['without_PV']['f']['samples'])

# Tmax
graphLabels([np.array(data['without_PV']['Tmax'][f'Time_{Tmax}'])/60 for Tmax in Tmax_vector], [data['without_PV']['Tmax'][f'LossProb_{Tmax}'] for Tmax in Tmax_vector], [f'Tmax: '+f'{Tmax}h' for Tmax in Tmax_vector], 'Loss Probability for different Tmax', 'Time of the day', 'Loss Probability', 'lossProb_Tmax_without_PV.pdf')

graphLabels([np.array(data['without_PV']['Tmax'][f'Time_{Tmax}'])/60 for Tmax in Tmax_vector], [data['without_PV']['Tmax'][f'Cost_{Tmax}'] for Tmax in Tmax_vector], [f'Tmax: '+f'{Tmax}h' for Tmax in Tmax_vector], 'Cost for different Tmax', 'Time of the day', 'Cumulative expenditure (euro/MWh)', 'cost_Tmax_without_PV.pdf')

# f
end = 11
graphLabels([np.array(data['without_PV']['f'][f'Time_{f}'])/60 for f in f_vector[:end]], [data['without_PV']['f'][f'LossProb_{f}'] for f in f_vector[:end]], [f'Fraction: '+f'{f*100}' for f in f_vector[:end]], 'Loss Probability for different fraction of non working batteries', 'Time of the day', 'Loss Probability', 'lossProb_f_without_PV.pdf')

graphLabels([np.array(data['without_PV']['f'][f'Time_{f}'])/60 for f in f_vector[:end]], [data['without_PV']['f'][f'Cost_{f}'] for f in f_vector[:end]], [f'Fraction: '+f'{f*100}' for f in f_vector[:end]], 'Cost for different fraction of non working batteries', 'Time of the day', 'Cumulative expenditure (euro/MWh)', 'cost_f_without_PV.pdf')

# Working batteries f
graphLabels([np.array(data['without_PV']['f'][f'Time_{f}'])/60 for f in f_vector], [data['without_PV']['f'][f'Working_{f}'] for f in f_vector], [f'Fraction: '+f'{f*100}' for f in f_vector], 'Number of working batteries for different fraction of non working batteries', 'Time of the day', 'Number of working charging stations', 'working_f_without_PV.pdf')

# %% with Solar Panels
# f
graphLabels([np.array(data['with_PV']['f'][f'Time_{f}'])/60 for f in f_vector[:end]], [data['with_PV']['f'][f'LossProb_{f}'] for f in f_vector[:end]], [f'Fraction: '+f'{f*100}' for f in f_vector[:end]], 'Loss Probability for different fraction of non working batteries', 'Time of the day', 'Loss Probability', 'lossProb_f_with_PV.pdf')

# Working batteries f
graphLabels([np.array(data['with_PV']['f'][f'Time_{f}'])/60 for f in f_vector], [data['with_PV']['f'][f'Working_{f}'] for f in f_vector], [f'Fraction: '+f'{f*100}' for f in f_vector], 'Number of working batteries for different fraction of non working batteries', 'Time of the day', 'Number of working charging stations', 'working_f_with_PV.pdf')