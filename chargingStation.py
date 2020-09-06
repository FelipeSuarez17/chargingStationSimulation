import random
import json
from queue import Queue, PriorityQueue
import numpy as np
import seaborn as sns
from scipy.stats import t, sem
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

C = 40  # Max battery capacity
NBSS = 5  # Max number of chargers
Wmax = 7  # Max waiting time for EV
Bth = 40  # Battery capacity
BthHighDemand = 40  # Accepted minimum charge level
deltaHighDemand = 60
lossesHighDemand = 2
chargingRate = 20  # charging rate per hour
maxChargingRate = 20  # Fixed charging rate
PV = 0  # Number of Photovoltaic Panels
S_one_PV = 1  # Nominal Cap. of one PV (1kWp)
prices = pd.read_csv('Data/electricity_prices.csv')  # Prices dataframe
PV_production = pd.read_csv('Data/PVproduction_PanelSize1kWp.csv')  # Output PV power dataframe
day = 1
month = 1
Spv = 0  # Nominal Cap. of the set of PV (kW), as we start at midnight the nom. cap. will always be 0


class Measure:
    def __init__(self):
        self.arr = 0
        self.dep = 0
        self.ut = 0
        self.oldT = 0
        self.loss = []
        self.cost = 0
        self.waitingTime = []
        self.chargingTime = []


class Battery:
    def __init__(self, arrival_time, charger, Bth=40, inStation=False):
        self.arrival_time = arrival_time
        self.bth = Bth
        if inStation:
            # self.level = batLevel(0, 2)
            # TODO implement k method to remove warm-up
            # TODO graph for different initial charging values the warm-up period
            self.level = batLevel(10, 1)
            self.estimateAvailable = self.arrival_time + ((Bth - self.level) * 60 / chargingRate)  # estimated waiting time for next available battery
            FES.put((self.estimateAvailable, "batteryAvailable", charger))
        else:
            self.level = batLevel(Bth / 4, 1)
        self.charger = charger


class Charger:
    def __init__(self, NBSS):
        self.chargers = []
        self.working = []
        for i in range(NBSS):
            self.chargers.append(Battery(arrival_time=0, charger=i, inStation=True, Bth=Bth))
            self.working.append(True)


def batLevel(mean, std):  # generate random initial charge level
    global Bth
    level = np.random.normal(mean, std)
    if 0 < level < Bth:
        return level
    else:
        return batLevel(mean, std)


def getNextArrival(time, fixed=False):
    """
    time:hour of the day to generate random interarrival time according the given distribution arrivalRateCoeff
    fixed: True if fixed average arrival rate for the EVs according fixed time fixedNextArrival
    """
    if fixed:
        nextArrival = 5
    else:
        arrivalRateCoeff = [30, 30, 30, 30, 20, 15, 13, 10, 5, 8, 15, 15, 3, 4, 10, 13, 15, 15, 2, 5, 15, 18, 20, 25]  # distribution arrival rate (mean time between arrivals)
        hour = int(time / 60)  # hour index from time in minutes
        nextArrival = random.expovariate(1 / arrivalRateCoeff[hour])  # generate arrival time in minutes as function of the hour
    return nextArrival  # minutes


def arrival(time, FES, waitingLine):
    global Bth
    data.arr += 1
    data.ut += len(waitingLine) * (time - data.oldT)
    if getLosses(data.loss, time, delta=deltaHighDemand) >= lossesHighDemand:  # define high demand even though the departure was already scheduled
        Bth = BthHighDemand
    else:
        Bth = 40
    inter_arrival = getNextArrival(time, False)  # get inter_arrival time, True for fixed time
    FES.put((time + inter_arrival, "arrival", -1))  # schedule the next arrival

    updateBatteriesLevel(time, data.oldT)  # updated each battery level in chargers(in station) and update cost

    estimatedWaitings = []
    for i in range(len(chargers.chargers)):
        estimatedWaitings.append(chargers.chargers[i].estimateAvailable)
    estimatedWaitings = np.sort(estimatedWaitings)

    if len(waitingLine) < NBSS:  # full waiting line
        residualWaiting = estimatedWaitings[len(waitingLine)] - time
        # if residualWaiting <= 0:
        #     consumptionTime = time - data.oldT + residualWaiting
        if 0 < residualWaiting < Wmax:
            waitingLine.append(Battery(arrival_time=time, charger=-1, Bth=Bth))  # charger=-1 means that battery is not charging yet
        elif residualWaiting <= 0:  # Battery available, then EV immediately served
            data.dep += 1
            oldBatteryEV = Battery(arrival_time=time, charger=-1, Bth=Bth)
            for i in range(len(chargers.chargers)):
                if chargers.chargers[i].estimateAvailable == estimatedWaitings[len(waitingLine)]:  # More than 1 battery charged, serve with equal available time
                    oldBatteryEV.charger = i
                    data.waitingTime.append(0)  # The car does not wait for the charged battery
                    data.chargingTime.append(time-chargers.chargers[i].arrival_time)  # Compute charging battery time
                    oldBatteryEV.estimateAvailable = time + (Bth - oldBatteryEV.level) * 60 / chargingRate
                    chargers.chargers[i] = oldBatteryEV  # replace battery in charger
                    # check_add_event(FES, i)  # Check if a charger has already an event
                    FES.put((oldBatteryEV.estimateAvailable, "batteryAvailable", i))
        else:
            data.loss.append(time)  # List of time when the loss occurred
            data.waitingTime.append(Wmax)
    else:  # loss
        data.loss.append(time)  # List of time when the loss occurred
        data.waitingTime.append(Wmax)

    data.oldT = time


def updateBatteriesLevel(time, oldT):  # update batteries level and cost
    global chargingRate, Spv
    deltaCharge = (time - oldT) * chargingRate / 60
    listCosts = getCosts(time, oldT)
    for i in range(len(chargers.chargers)):
        # if chargers.chargers[i].level + deltaCharge > C:
        #     chargers.chargers[i].level = C  # when battery is fully charged
        # else:
        #     chargers.chargers[i].level = chargers.chargers[i].level + deltaCharge  # update battery level
        if chargers.working[i]:  # Check if chargers are working (work postponed due to high cost, daylight, and Tmax)
            if chargers.chargers[i].level != C:  # If battery is not charged at full capacity
                chargers.chargers[i].level = chargers.chargers[i].level + deltaCharge  # update battery level
                if Spv == 0:  # If Spv is equal to 0 it means we are using the power grid and we need to pay for that
                    for pair in listCosts:
                        if (time - oldT) != 0:  # avoid zero division
                            data.cost += ((pair[0]) / (time - oldT)) * deltaCharge * pair[1]  # adding the cost of the fraction of time according to listCosts


def batteryAvailable(time, FES, waitingLine, charger):  # departure
    global Bth
    updateBatteriesLevel(time, data.oldT)  # updated each battery level in chargers(in station) and update cost
    data.ut += len(waitingLine) * (time - data.oldT)
    if getLosses(data.loss, time, delta=deltaHighDemand) >= lossesHighDemand:
        Bth = BthHighDemand
    else:
        Bth = 40

    if len(waitingLine) != 0:
        data.dep += 1
        oldBatteryEV = waitingLine.pop(0)  # take battery from car
        data.waitingTime.append(time-oldBatteryEV.arrival_time)  # To estimate the individual waiting time
        data.chargingTime.append(time-chargers.chargers[charger].arrival_time)  # To estimate the charging battery time
        # newBatteryEV = chargers.chargers[i]
        oldBatteryEV.charger = charger
        oldBatteryEV.arrival_time = time
        oldBatteryEV.estimateAvailable = time + (Bth - oldBatteryEV.level) * 60 / chargingRate
        chargers.chargers[charger] = oldBatteryEV  # replace battery in charger
        FES.put((oldBatteryEV.estimateAvailable, "batteryAvailable", charger))
    data.oldT = time


def updateEstimateAvailable(time):
    for i in range(len(chargers.chargers)):
        if chargers.working[i]:
            if (Bth - chargers.chargers[i].level) != 0:  # If battery is already charged we don´t change the estimate available charge time
                chargers.chargers[i].estimateAvailable = time + (Bth - chargers.chargers[i].level) * 60 / chargingRate
                j = 0
                while j < len(FES.queue):
                    if FES.queue[j][1] in 'batteryAvailable' and FES.queue[j][2] == i:
                        FES.queue.pop(j)
                        break
                    else:
                        j += 1
                FES.put((chargers.chargers[i].estimateAvailable, 'batteryAvailable', i))
            # TODO chargingRate graph depending the hour and compare it with the different seasons, daylight duration and chargingRate
    # Check waiting line (remove if necessary)
    estimatedWaitings = []
    for i in range(len(chargers.chargers)):
        if chargers.working[i]:
            estimatedWaitings.append(chargers.chargers[i].estimateAvailable)
    estimatedWaitings = np.sort(estimatedWaitings)
    i = 0
    while i < len(waitingLine):
        if waitingLine[i].arrival_time + Wmax <= estimatedWaitings[i]:  # If EV needs to wait more than the arrival time plus Wmax, the EV leaves and is added to losses
            waitingLine.pop(i)
            data.loss.append(time)
            data.waitingTime.append(Wmax)
        else:
            i += 1


def chargingRate_change(time, FES):
    global chargingRate, Spv
    updateBatteriesLevel(time, data.oldT)  # updated each battery level in chargers (in station) and update cost
    FES.put((time + 60, "chargingRate_change", -1))  # Check every hour
    hour = int(time / 60)
    if hour % 24 == 0:  # Bug fix when there is a new day we start again at hour equals 0 not 24
        hour = 0
    PowerDayHour = PV_production[(PV_production['Month'] == month) & (PV_production['Day'] == day) & (PV_production['Hour'] == hour)]
    OutPow = PowerDayHour.iloc[0][3]  # Retrieving output power according to day, month, and hour
    Spv = S_one_PV * PV * OutPow  # Power of a set of panels in a given day, month, and hour (Wh)
    if Spv == 0:  # If solar panels are not producing energy (night hours)
        chargingRate = maxChargingRate
        # if checkHighCost(hour)
    else:
        chargingRate = (Spv/NBSS)/1000  # Over 1000 to convert it to kWh
        if chargingRate > 20:  # if charging rate is more than 20 kWh limit that power to avoid battery damage
            chargingRate = 20
    updateEstimateAvailable(time)
    data.oldT = time


def getCosts(time, oldT):  # return eur/kWh
    cost = []
    season = getSeason()
    if int(oldT / 60) == int(time / 60):
        iterHour = int(time / 60)
        iterCost = prices[(prices['Hour'] == iterHour) & (prices['Season'] == season)].iloc[0]['Cost'] / 1000
        cost.append((time - oldT, iterCost))
    else:
        iterHour = int(oldT / 60)
        iterCost = prices[(prices['Hour'] == iterHour) & (prices['Season'] == season)].iloc[0]['Cost'] / 1000
        cost.append((60 - (oldT % 60), iterCost))
        iterHour += 1
        while iterHour != int(time / 60):
            if iterHour == 24:
                iterHour = 0
            iterCost = prices[(prices['Hour'] == iterHour) & (prices['Season'] == season)].iloc[0]['Cost'] / 1000
            cost.append((60, iterCost))
            iterHour += 1
        if iterHour == 24:
            iterHour = 0
        iterCost = prices[(prices['Hour'] == iterHour) & (prices['Season'] == season)].iloc[0]['Cost'] / 1000
        cost.append((time % 60, iterCost))
    return cost


def getSeason():
    if datetime(2020, 3, 20) <= datetime(2020, month, day) < datetime(2020, 6, 20):
        season = "SPRING"
    elif datetime(2020, 6, 20) <= datetime(2020, month, day) < datetime(2020, 9, 22):
        season = "SUMMER"
    elif datetime(2020, 9, 22) <= datetime(2020, month, day) < datetime(2020, 12, 21):
        season = "FALL"
    elif (datetime(2020, 1, 1) <= datetime(2020, month, day) < datetime(2020, 3, 20)) or (datetime(2020, 12, 21) <= datetime(2020, month, day) <= datetime(2020, 12, 31)):
        season = "WINTER"
    else:
        season = "NA"
    return season


def getLosses(lossses_list, time, delta):
    count = 0
    for i in lossses_list:
        if time - delta < i < time:  # Count the number of losses in the time span defined by the delta
            count += 1
    return count


def plotCDF(data, xlabel, ylabel, name):
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    plt.plot(data_sorted, p)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([min(data) - 1, max(data)])
    # plt.savefig(name)
    plt.show()


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    SIM_TIME = 24 * 60  # Simulation time in minutes
    time = 0
    waitingLine = []
    data = Measure()
    FES = PriorityQueue()  # list of events
    FES.put((0, "arrival", -1))  # schedule first arrival at t=0
    FES.put((60, "chargingRate_change", -1))
    chargers = Charger(NBSS)
    listTime = []
    listChargingRate = []
    pbar = tqdm(total=SIM_TIME)
    while time < SIM_TIME:
        (time, event_type, charger) = FES.get()

        if event_type == "arrival":
            arrival(time, FES, waitingLine)

        elif event_type == "batteryAvailable":
            batteryAvailable(time, FES, waitingLine, charger)

        elif event_type == "chargingRate_change":
            chargingRate_change(time, FES)
        listTime.append(time)
        listChargingRate.append(chargingRate)
        pbar.update(time)
    confidence_int_wait = t.interval(0.999, len(data.waitingTime) - 1, np.mean(data.waitingTime), sem(data.waitingTime))
    confidence_int_charge = t.interval(0.999, len(data.chargingTime) - 1, np.mean(data.chargingTime), sem(data.chargingTime))
    # Loss warm-up period (hour 0-3)
    warm_loss = np.count_nonzero(np.array(data.loss) < 180)
    print(f"Confidence interval Waiting Time: {confidence_int_wait}")
    print(f"Confidence interval Charging Time: {confidence_int_charge}")
    print(f"Number of arrivals: {data.arr}")
    print(f"Number of departures: {data.dep}")
    print(f"Number of losses: {len(data.loss)}")
    print(f"Number of losses in the warm-up interval: {warm_loss}")
    # TODO compute the average waiting delay for k minus samples 
    plotCDF(data.loss, "", "", "test.pdf")
    plt.figure()
    plt.plot(data.waitingTime)
    plt.show()
