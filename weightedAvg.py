import json
import numpy as np

data = json.load(open('data.json'))
config = json.load(open('config.json'))

# %%

# Extracting all data

for key in config.keys():
    for season in ['winter', 'summer']:
        f_vector = np.linspace(config[key]['f']['start'], config[key]['f']['finish'], config[key]['f']['samples'])
        Tmax_vector = np.linspace(config[key]['Tmax']['start'], config[key]['Tmax']['finish'], config[key]['Tmax']['samples'])
        for Tmax in Tmax_vector:
            all_seedsTime = []
            all_seedsLossProb = []
            all_seedsCost = []
            all_seedsWorking = []
            [all_seedsTime.append(np.array(data[key]['Tmax'][f'Time_{Tmax}_{seed}_{season}']) / 60) for seed in range(50)]
            [all_seedsLossProb.append(np.array(data[key]['Tmax'][f'LossProb_{Tmax}_{seed}_{season}'])) for seed in range(50)]
            [all_seedsCost.append(np.array(data[key]['Tmax'][f'Cost_{Tmax}_{seed}_{season}'])) for seed in range(50)]
            [all_seedsWorking.append(np.array(data[key]['Tmax'][f'Working_{Tmax}_{seed}_{season}'])) for seed in range(50)]
            # Tuple Time-LossProb
            config[key]['Tmax'][f'Time_LossProb_{Tmax}_{season}'] = sorted(zip(np.concatenate(all_seedsTime).tolist(), np.concatenate(all_seedsLossProb).tolist()))
            config[key]['Tmax'][f'Time_{Tmax}_{season}'] = [element for element, _ in config[key]['Tmax'][f'Time_LossProb_{Tmax}_{season}']]
            config[key]['Tmax'][f'allLossProb_{Tmax}_{season}'] = [element for _, element in config[key]['Tmax'][f'Time_LossProb_{Tmax}_{season}']]
            # Tuple Time-Costs
            config[key]['Tmax'][f'Time_Cost_{Tmax}_{season}'] = sorted(zip(np.concatenate(all_seedsTime).tolist(), np.concatenate(all_seedsCost).tolist()))
            config[key]['Tmax'][f'allCost_{Tmax}_{season}'] = [element for _, element in config[key]['Tmax'][f'Time_Cost_{Tmax}_{season}']]
            # Tuple Time-Working
            config[key]['Tmax'][f'Time_Working_{Tmax}_{season}'] = sorted(zip(np.concatenate(all_seedsTime).tolist(), np.concatenate(all_seedsWorking).tolist()))
            config[key]['Tmax'][f'allWork_{Tmax}_{season}'] = [element for _, element in config[key]['Tmax'][f'Time_Working_{Tmax}_{season}']]
        for f in f_vector:
            all_seedsTime = []
            all_seedsLossProb = []
            all_seedsCost = []
            all_seedsWorking = []
            [all_seedsTime.append(np.array(data[key]['f'][f'Time_{f}_{seed}_{season}']) / 60) for seed in range(50)]
            [all_seedsLossProb.append(np.array(data[key]['f'][f'LossProb_{f}_{seed}_{season}'])) for seed in range(50)]
            [all_seedsCost.append(np.array(data[key]['f'][f'Cost_{f}_{seed}_{season}'])) for seed in range(50)]
            [all_seedsWorking.append(np.array(data[key]['f'][f'Working_{f}_{seed}_{season}'])) for seed in range(50)]
            # Tuple Time-LossProb
            config[key]['f'][f'Time_LossProb_{f}_{season}'] = sorted(zip(np.concatenate(all_seedsTime).tolist(), np.concatenate(all_seedsLossProb).tolist()))
            config[key]['f'][f'Time_{f}_{season}'] = [element for element, _  in config[key]['f'][f'Time_LossProb_{f}_{season}']]
            config[key]['f'][f'allLossProb_{f}_{season}'] = [element for _, element in config[key]['f'][f'Time_LossProb_{f}_{season}']]
            # Tuple Time-Costs
            config[key]['f'][f'Time_Cost_{f}_{season}'] = sorted(zip(np.concatenate(all_seedsTime).tolist(), np.concatenate(all_seedsCost).tolist()))
            config[key]['f'][f'allCost_{f}_{season}'] = [element for _, element in config[key]['f'][f'Time_Cost_{f}_{season}']]
            # Tuple Time-Working
            config[key]['f'][f'Time_Working_{f}_{season}'] = sorted(zip(np.concatenate(all_seedsTime).tolist(), np.concatenate(all_seedsWorking).tolist()))
            config[key]['f'][f'allWork_{f}_{season}'] = [element for _, element in config[key]['f'][f'Time_Working_{f}_{season}']]
        # Exponentially weighted moving average
        beta = 0.999
        for Tmax in Tmax_vector:
            weighted_lossProb = []
            weigthed_cost = []
            weigthed_work =[]
            loss = 1
            cost = 0
            work = 10
            i = 0
            while i < len(config[key]['Tmax'][f'allLossProb_{Tmax}_{season}']):
                loss = beta*loss + (1-beta)*config[key]['Tmax'][f'allLossProb_{Tmax}_{season}'][i]
                cost = beta*cost + (1-beta)*config[key]['Tmax'][f'allCost_{Tmax}_{season}'][i]
                work = beta * work + (1 - beta) * config[key]['Tmax'][f'allWork_{Tmax}_{season}'][i]
                # Bias correction
                # loss = loss/(1-beta**(i+1))
                # cost = cost/(1-beta**(i+1))
                weighted_lossProb.append(loss)
                weigthed_cost.append(cost)
                weigthed_work.append(work)
                i += 1
            config[key]['Tmax'][f'LossProb_{Tmax}_{season}'] = weighted_lossProb
            config[key]['Tmax'][f'Cost_{Tmax}_{season}'] = weigthed_cost
            config[key]['Tmax'][f'Working_{Tmax}_{season}'] = weigthed_work
        for f in f_vector:
            weighted_lossProb = []
            weigthed_cost = []
            weigthed_work = []
            loss = 1
            cost = 0
            work = 10
            i = 0
            while i < len(config[key]['f'][f'allLossProb_{f}_{season}']):
                loss = beta*loss + (1-beta)*config[key]['f'][f'allLossProb_{f}_{season}'][i]
                cost = beta*cost + (1-beta)*config[key]['f'][f'allCost_{f}_{season}'][i]
                work = beta * work + (1 - beta) * config[key]['f'][f'allWork_{f}_{season}'][i]
                # Bias correction
                # loss = loss / (1 - beta ** (i+1))
                # cost = cost / (1 - beta ** (i+1))
                weighted_lossProb.append(loss)
                weigthed_cost.append(cost)
                weigthed_work.append(work)
                i += 1
            config[key]['f'][f'LossProb_{f}_{season}'] = weighted_lossProb
            config[key]['f'][f'Cost_{f}_{season}'] = weigthed_cost
            config[key]['f'][f'Working_{f}_{season}'] = weigthed_work
json_data = json.dumps(config, indent=4)
with open('data_weighted.json', 'w') as outfile:
    outfile.write(json_data)
