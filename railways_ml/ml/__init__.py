import xgboost as xgb
import numpy as np
from sklearn import preprocessing
import pandas as pd
import datetime


def runML(date, carClass, trainNumber, stationName, indexNumber):
    ticket_reg = xgb.XGBRegressor(max_depth=4, n_estimators=100, random_state=0)
    ticket_reg.load_model('./railways_ml/ml/ticket.bin')

    count_reg = xgb.XGBRegressor(max_depth=4, n_estimators=100, random_state=0)
    count_reg.load_model('./railways_ml/ml/count.bin')

    station_le = preprocessing.LabelEncoder()
    train_le = preprocessing.LabelEncoder()
    station_le.classes_ = np.load('./railways_ml/ml/StationName.npy', allow_pickle=True)
    train_le.classes_ = np.load('./railways_ml/ml/TrainNumber.npy', allow_pickle=True)

    data = pd.DataFrame(
        columns=['StationName', 'IndexNumber', 'TrainNumber', 'WeekDay', 'Month', 'IsWeekend', '1Д', '1Л', '2Д', '2К',
                 '2Л', '2С', '3О', '3П'])
    data['TrainNumber'] = train_le.transform(trainNumber)
    data['StationName'] = station_le.transform(stationName)
    data['IndexNumber'] = indexNumber
    data['DepartureDate'] = date
    data['WeekDay'] = data['DepartureDate'].apply(
        lambda x: datetime.datetime(int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2])).weekday())
    data['Month'] = data['DepartureDate'].apply(
        lambda x: datetime.datetime(int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2])).month)
    data['IsWeekend'] = data.WeekDay.apply(lambda x: 1 if x > 4 else 0)
    data = data.drop(columns='DepartureDate')

    for col in data.columns:
        if len(col) == 2:
            data[col] = np.zeros(len(date))

    for idx, carCls in enumerate(carClass):
        data.iloc[idx][carCls] = 1

    preds = {}
    preds['Count'] = np.maximum(np.zeros(len(data)), count_reg.predict(data))
    preds['TicketsSold'] = np.maximum(np.zeros(len(data)), ticket_reg.predict(data))
    preds['TicketsSold'] = np.maximum(preds['TicketsSold'], preds['Count'])
    for i in range(1, len(preds['TicketsSold'])):
        preds['TicketsSold'][i] = max(preds['TicketsSold'][i], preds['TicketsSold'][i - 1])
    return preds
