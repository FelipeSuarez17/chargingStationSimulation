import random
import json
from queue import Queue, PriorityQueue
import numpy as np
import seaborn as sns
from scipy.stats import t, sem
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

C = 40  # Max battery capacity
NBSS = 5  # Max number of chargers
Wmax = 7  # Max waiting time for EV
Bth = 40  # Accepted minimum charge level
BthHighDemand = 20
deltaHighDemand = 60
lossesHighDemand = 2
chargingRate = 20  # charging rate per hour
prices = pd.read_csv('Data/electricity_prices.csv')  # Prices dataframe
day = 1
month = 1


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
            self.level = batLevel(35, 5)
            self.estimateAvailable = self.arrival_time + ((Bth - self.level) * 60 / chargingRate)  # estimated waiting time for next available battery
            FES.put((self.estimateAvailable, "batteryAvailable", charger))
        else:
            self.level = batLevel(Bth / 4, 1)
        self.charger = charger


class Charger:
    def __init__(self, NBSS):
        self.chargers = []
        for i in range(NBSS):
            self.chargers.append(Battery(arrival_time=0, charger=i, inStation=True, Bth=Bth))


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
        arrivalRateCoeff = [30, 30, 30, 30, 20, 15, 13, 10, 5, 8, 15, 15, 3, 4, 10, 13, 15, 15, 2, 5, 15, 18, 20, 25]  # distribution arrival rate
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
    inter_arrival = getNextArrival(time, True)  # get inter_arrival time, True for fixed time
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
            data.loss.append(time)
            data.waitingTime.append(7)
    else:  # loss
        data.loss.append(time)
        data.waitingTime.append(7)

    data.oldT = time


def updateBatteriesLevel(time, oldT):  # update batteries level and cost
    deltaCharge = (time - oldT) * chargingRate / 60
    listCosts = getCosts(time, oldT)
    for i in range(len(chargers.chargers)):
        # if chargers.chargers[i].level + deltaCharge > C:
        #     chargers.chargers[i].level = C  # when battery is fully charged
        # else:
        #     chargers.chargers[i].level = chargers.chargers[i].level + deltaCharge  # update battery level
        if chargers.chargers[i].level != C:
            chargers.chargers[i].level = chargers.chargers[i].level + deltaCharge  # update battery level
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
        if time - delta < i < time:
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
    SIM_TIME = 24 * 60  # Simulation time
    time = 0
    waitingLine = []
    data = Measure()
    FES = PriorityQueue()  # list of events
    FES.put((0, "arrival", -1))  # schedule first arrival at t=0
    chargers = Charger(NBSS)
    while time < SIM_TIME:
        (time, event_type, charger) = FES.get()
        # print(time)

        if event_type == "arrival":
            arrival(time, FES, waitingLine)

        elif event_type == "batteryAvailable":
            batteryAvailable(time, FES, waitingLine, charger)
    # data.waitingTime = data.waitingTime + [Wmax for i in range(len(data.loss))]  # Add the waiting time of losses
    confidence_int_wait = t.interval(0.999, len(data.waitingTime)-1, np.mean(data.waitingTime), sem(data.waitingTime))
    confidence_int_charge = t.interval(0.999, len(data.chargingTime) - 1, np.mean(data.chargingTime), sem(data.chargingTime))
    print(f"Confidence interval Waiting Time: {confidence_int_wait}")
    print(f"Confidence interval Charging Time: {confidence_int_charge}")
    print(f"Number of arrivals: {data.arr}")
    print(f"Number of departures: {data.dep}")
    print(f"Number of losses: {len(data.loss)}")
    plotCDF(data.loss, "", "", "test.pdf")
    # TODO include PV
