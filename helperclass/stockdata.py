import bs4 as bs
from datetime import date, datetime
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import requests
import yfinance as yf


class StockData:

    # Initialize
    def __init__(self, start="2010-01-01", end=date.today().strftime('%Y-%m-%d')):
        self.start_time = start
        self.end_time = end
        self.train_start = None
        self.train_end = None
        self.test_end = None
        self.test_stock_list=None
        self.stocks_label = self.__get_stocks_label()
        self.alltime_stock_data = self.__get_all_time_stock_data()

    # Function
    def __get_all_time_stock_data(self):
        try:
            prices_raw = pd.read_csv('./helperData/prices_raw.csv', index_col=0)

            last_trade_day = prices_raw.index[-1]
            delta = datetime(date.today().year, date.today().month, date.today().day) - datetime(
                int(last_trade_day[0:4]), int(last_trade_day[5:7]), int(last_trade_day[8:]))
            int_delta = delta.days
            self.stocks_label = prices_raw.columns.to_list()

            if int_delta >= 40:
                raise FileNotFoundError

        except FileNotFoundError:
            resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            soup = bs.BeautifulSoup(resp.text, 'lxml')
            table = soup.find('table', {'class': 'wikitable sortable'})

            tickers = []
            for row in table.findAll('tr')[1:]:
                ticker = row.findAll('td')[0].text
                tickers.append(ticker)

            stocks = [s.replace('\n', '') for s in tickers]
            exclusion_list = ["VLTO", "BF.B", "BRK.B"]
            stocks = list(set(stocks) - set(exclusion_list))
            stocks.sort()

            yf.pdr_override()

            # Load all stock price data
            prices_raw = pdr.get_data_yahoo(stocks, start=self.start_time, end=self.end_time)[['Adj Close']]
            prices_raw = prices_raw['Adj Close'][stocks]

            # Add cash option
            prices_raw["Cash"] = 1
            self.stocks_label = prices_raw.columns.to_list()

            prices_raw.to_csv("./helperData/prices_raw.csv", index=True)

        return prices_raw

    def __get_stocks_label(self):
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})

        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text
            tickers.append(ticker)

        stocks = [s.replace('\n', '') for s in tickers]
        exclusion_list = ["VLTO", "BF.B", "BRK.B"]
        stocks = list(set(stocks) - set(exclusion_list))
        stocks.sort()

        return stocks

    def get_stock_data(self):

        temp = self.alltime_stock_data.copy()
        temp.dropna(axis=1, inplace=True)


        # temp.interpolate(inplace=True, limit_direction="both")
        X = temp.loc[(temp.index >= self.train_start) & (temp.index <= self.test_end)].apply(np.log).apply(np.diff)
        X_train = temp.loc[(temp.index >= self.train_start) & (temp.index <= self.train_end)].apply(np.log).apply(np.diff)
        X_train = X.iloc[:len(X_train)].reset_index(drop=True)
        X_test = X.iloc[len(X_train):].reset_index(drop=True)

        
        # X_train = temp.loc[(temp.index >= self.train_start) & (temp.index <= self.train_end)].apply(np.log).apply(np.diff)
        #X_test = temp.loc[(temp.index > self.train_end) & (temp.index <= self.test_end)].apply(np.log).apply(np.diff)
        self.stocks_label = X.columns.to_list()
        
        if self.test_stock_list:
            X=X[self.test_stock_list]
            X_train=X_train[self.test_stock_list]
            X_test=X_test[self.test_stock_list]

            self.stocks_label = self.test_stock_list

        return X, X_train, X_test
