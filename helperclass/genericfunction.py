import pandas as pd
from mosek.fusion import *
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
import pickle
from datetime import datetime
from datetime import timedelta
from random import randrange
from geom_median.numpy import compute_geometric_median
from helperclass import database


class GenericFunction:

    # Initialize
    def __init__(self, stockdata):
        self.stockdata = stockdata
        self.n_day_rebalance = 1
        self.ssr_properties = [0, 0]
        self.ensemble_properties = [0, 0, 0, 0]
        self.boot_properties = 50
        self.gamma2 = 0.001
        self.aggregate_method = "sample_mean"
        self.returns = None
        self.sharpe = None
        self.vo = None
        self.to = None
        self.mdd = None
        self.disable_tqdm = False

    # Equal-weighted Portfolio
    def __ewp(self, print_status=True):
        X, X_train, X_test = self.stockdata.get_stock_data()

        X_test = X_test.to_numpy()
        weights = self.generic_rolling_window_helper("ewp", X, len(X_train), self.n_day_rebalance)
        returns = 1 + (X_test * weights).sum(axis=1)

        # Calculate Sharpe Ratio
        sharpe = np.mean((returns - 1)) / np.std((returns - 1)) * np.sqrt(252)

        # Calculate Volatility
        vo = np.sqrt(252) * np.std(returns)

        # Calculate Turnover Rate
        total = 0
        for k in range(len(X_test) - 2):  # loop to M-2
            # Calculate difference between weights at k and weights at k+1
            total += TO(X_test[k]+1, weights[k], weights[k + 1])
        to = total / (len(X_test) - 1)  # M-R-1

        # Calculate Maximum Drawdown
        cum_ret = returns.cumprod()
        cum_ret = pd.Series(cum_ret)
        window = 252
        Roll_Max = cum_ret.rolling(window, min_periods=1).max()
        Daily_Drawdown = cum_ret / Roll_Max - 1.0
        Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
        mdd = Max_Daily_Drawdown.min()*-1

        # For calculation
        cum_ret = cum_ret - 1

        self.returns = returns
        self.sharpe = sharpe
        self.vo = vo
        self.to = to
        self.mdd = mdd

        # Import to database
        run_id = "Run" + database.get_last_id()
        data_to_weights = []
        for i in range(len(weights)):
            each_day_weights = []
            for j in range(weights.shape[1]):
                each_day_weights.append(weights[i][j])
            data_to_weights.append((run_id, "Day" + str(i + 1), str(each_day_weights)))

        data_to_details = [(run_id, "EWP", self.gamma2, self.aggregate_method, self.n_day_rebalance, self.stockdata.train_start, self.stockdata.train_end,
                            self.stockdata.test_end, cum_ret.iloc[-1], sharpe, vo, to, mdd, "NA", "NA", "NA", "NA",
                            "NA", "NA", len(self.stockdata.stocks_label), str(self.stockdata.stocks_label))]

        database.append_to_weights(data_to_weights)
        database.append_to_details(data_to_details)

        if print_status:
            print("Run details have been saved.")
            print(f"EWP || return: {cum_ret.iloc[-1]}, sharpe: {sharpe}, Volatility: {vo}, Turnover Rate: {to}, Maximum Drawdown: {mdd}")

    # Mean-variance Portfolio
    def __mvp(self, X):  # 20240308
        with Model("markowitz") as M:
            N = X.shape[1]

            try:
                Sigma = LedoitWolf().fit(X.to_numpy()).covariance_
                m = X.mean().to_numpy()
            except:
                Sigma = LedoitWolf().fit(X).covariance_
                m = np.mean(X, axis=0)
            # Use LW shrinkage to make Sigma PD
            G = np.linalg.cholesky(Sigma)

            # No short-selling
            x = M.variable("x", N, Domain.greaterThan(0.0))

            # Objective
            M.objective('obj', ObjectiveSense.Maximize, Expr.dot(m, x))

            # Budget constraint
            M.constraint('budget', Expr.sum(x), Domain.equalsTo(1))

            # Imposes a bound on the risk
            M.constraint('risk', Expr.vstack(self.gamma2, 0.5, Expr.mul(G.transpose(), x)), Domain.inRotatedQCone())

            # Solve optimization
            M.solve()

            # Check if the solution is an optimal point
            solsta = M.getPrimalSolutionStatus()
            if (solsta != SolutionStatus.Optimal):
                # See https://docs.mosek.com/latest/pythonfusion/accessing-solution.html about handling solution statuses.
                raise Exception("Unexpected solution status!")

            portfolio = x.level()

        return portfolio

    def __mvp_metrics(self, print_status=True):
        X, X_train, X_test = self.stockdata.get_stock_data()

        X_test = X_test.to_numpy()

        weights = self.generic_rolling_window_helper("mvp", X, len(X_train), self.n_day_rebalance)
        returns = 1+(X_test * weights).sum(axis=1)

        # Calculate Sharpe Ratio
        sharpe = np.mean(returns-1) / np.std(returns-1) * np.sqrt(252)

        # Calculate Volatility
        vo = np.sqrt(252) * np.std(returns)

        # Calculate Turnover Rate
        total = 0
        for k in range(len(X_test) - 2):  # loop to M-2
            # Calculate difference between weights at k and weights at k+1
            total += TO(X_test[k]+1, weights[k], weights[k + 1])
        to = total / (len(X_test) - 1)  # M-R-1

        # Calculate Maximum Drawdown
        cum_ret = returns.cumprod()
        cum_ret = pd.Series(cum_ret)
        window = 252
        Roll_Max = cum_ret.rolling(window, min_periods=1).max()
        Daily_Drawdown = cum_ret / Roll_Max - 1.0
        Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
        mdd = Max_Daily_Drawdown.min()*-1

        # For calculation
        cum_ret = cum_ret - 1

        self.returns = returns
        self.sharpe = sharpe
        self.vo = vo
        self.to = to
        self.mdd = mdd

        # Import to database
        run_id = "Run" + database.get_last_id()
        data_to_weights = []
        for i in range(len(weights)):
            each_day_weights = []
            for j in range(weights.shape[1]):
                each_day_weights.append(weights[i][j])
            data_to_weights.append((run_id, int(i + 1), str(each_day_weights)))

        data_to_details = [(run_id, "MVP", self.gamma2, self.aggregate_method, self.n_day_rebalance, self.stockdata.train_start, self.stockdata.train_end,
                            self.stockdata.test_end, cum_ret.iloc[-1], sharpe, vo, to, mdd, "NA", "NA", "NA", "NA",
                            "NA", "NA", len(self.stockdata.stocks_label), str(self.stockdata.stocks_label))]

        database.append_to_weights(data_to_weights)
        database.append_to_details(data_to_details)

        if print_status:
            print("Run details have been saved.")
            print(f"Traditional Markowitz (No rebalancing) || return: {cum_ret.iloc[-1]}, sharpe: {sharpe}, Volatility: {vo}, Turnover Rate: {to}, Maximum Drawdown: {mdd}")

    # SSR Algorithm
    """
    Parameters:
    s: number of sampled subsets
    b: number of assets in each sample
    i: time offset index for rolling window
    """
    def __ssr(self, X, s, b, base="mvp"):
        T_train, N_train = X.shape  # number of days, number of stocks
        stocks = self.stockdata.stocks_label  # stock list

        w_array = np.empty((0, N_train))
        for j in range(s):
            # Randomly sample a set Ij of b indices from all stocks without replacement
            Ij = random.sample(stocks, b)
            indices = [stocks.index(element) for element in Ij]  # indices of sampled stocks

            # Select the associated return data
            Rj = X[Ij]

            # Compute the optimal subset portfolio weights by MVP
            if base == "go":
                w = self.__growth_optimal_portfolio(Rj.T + 1)
            else:
                w = self.__mvp(Rj)
            w_dict = {key: value for key, value in zip(indices, w)}

            # Assign weights to stocks in sample, 0 otherwise
            w = [w_dict[i] if i in w_dict.keys() else 0 for i in range(N_train)]
            w_array = np.vstack((w_array, w))

        # Aggregate the constructed portfolio weights based on all resamples
        if self.aggregate_method == "sample_mean":
            w = w_array.mean(axis=0)
        else:
            w = compute_geometric_median(w_array, weights=None).median
        return w

    def __ssr_metrics(self, s, b, base, print_status=True):
        X, X_train, X_test = self.stockdata.get_stock_data()
        T_test, N_test = X_test.shape  # number of days, number of stocks
        X_test = X_test.to_numpy()

        if base == "mvp":
            weights = self.generic_rolling_window_helper("ssr_mvp", X, len(X_train), self.n_day_rebalance)
        elif base == "go":
            weights = self.generic_rolling_window_helper("ssr_go", X, len(X_train), self.n_day_rebalance)

        returns = 1 + (X_test * weights).sum(axis=1)

        # Calculate Sharpe Ratio
        sharpe = np.mean(returns-1) / np.std(returns-1) * np.sqrt(252)

        # Calculate Volatility
        vo = np.sqrt(252) * np.std(returns)  #20240311

        # Calculate Turnover Rate
        total = 0
        for k in range(T_test - 2):  # loop to M-2
            # Calculate difference between weights at k and weights at k+1
            total += TO(X_test[k]+1, weights[k], weights[k + 1])
        to = total / (T_test - 1)  # M-R-1

        # Calculate Maximum Drawdown
        cum_ret = returns.cumprod()
        cum_ret = pd.Series(cum_ret)
        window = 252
        Roll_Max = cum_ret.rolling(window, min_periods=1).max()
        Daily_Drawdown = cum_ret / Roll_Max - 1.0
        Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
        mdd = Max_Daily_Drawdown.min()*-1

        # For calculation
        cum_ret = cum_ret - 1

        self.returns = returns
        self.sharpe = sharpe
        self.vo = vo
        self.to = to
        self.mdd = mdd

        # Import to database
        run_id = "Run" + database.get_last_id()
        data_to_weights = []
        for i in range(len(weights)):
            each_day_weights = []
            for j in range(weights.shape[1]):
                each_day_weights.append(weights[i][j])
            data_to_weights.append((run_id, int(i + 1), str(each_day_weights)))

        data_to_details = [(run_id, "SSR_"+base, self.gamma2, self.aggregate_method, self.n_day_rebalance, self.stockdata.train_start, self.stockdata.train_end,
                            self.stockdata.test_end, cum_ret.iloc[-1], sharpe, vo, to, mdd, s, b, "NA", "NA",
                            "NA", "NA", len(self.stockdata.stocks_label), str(self.stockdata.stocks_label))]

        database.append_to_weights(data_to_weights)
        database.append_to_details(data_to_details)

        if print_status:
            print("Run details have been saved.")
            print(f"s = {s}, b = {b} || return: {cum_ret.iloc[-1]}, sharpe: {sharpe}, Volatility: {vo}, Turnover Rate: {to}, Maximum Drawdown: {mdd}")

    def __boot_metrics(self, base, print_status=True):
        X, X_train, X_test = self.stockdata.get_stock_data()
        T_test, N_test = X_test.shape  # number of days, number of stocks
        X_test = X_test.to_numpy()
        N = X.shape[0]
        samples = []
        k = self.boot_properties
        for _ in range(k):
            result_df=X_train.sample(N, replace=True, axis=0).reset_index(drop=True).copy()
            samples.append(result_df)
        
        weights_arr = []
        for sample in samples:
            if base == "mvp":
                weights = self.generic_rolling_window_helper("mvp", sample, len(X_train), self.n_day_rebalance)
            elif base == "go":
                weights = self.generic_rolling_window_helper("kelly", sample, len(X_train), self.n_day_rebalance)
            weights_arr.append(weights)

        # Aggregate portfolio weights
        if self.aggregate_method == "sample_mean":
            weights = np.mean(weights_arr, axis=0)
        else:
            weights = compute_geometric_median(weights_arr, weights=None).median

        returns = 1 + (X_test * weights).sum(axis=1)

        # Calculate Sharpe Ratio
        sharpe = np.mean(returns-1) / np.std(returns-1) * np.sqrt(252)

        # Calculate Volatility
        vo = np.sqrt(252) * np.std(returns)

        # Calculate Turnover Rate
        total = 0
        for k in range(T_test - 2):  # loop to M-2
            # Calculate difference between weights at k and weights at k+1
            total += TO(X_test[k]+1, weights[k], weights[k + 1])
        to = total / (T_test - 1)  # M-R-1

        # Calculate Maximum Drawdown
        cum_ret = returns.cumprod()
        cum_ret = pd.Series(cum_ret)
        window = 252
        Roll_Max = cum_ret.rolling(window, min_periods=1).max()
        Daily_Drawdown = cum_ret / Roll_Max - 1.0
        Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
        mdd = Max_Daily_Drawdown.min()*-1
        # For calculation
        cum_ret = cum_ret - 1

        self.returns = returns
        self.sharpe = sharpe
        self.vo = vo
        self.to = to
        self.mdd = mdd

        # Import to database
        run_id = "Run" + database.get_last_id()
        data_to_weights = []
        for i in range(len(weights)):
            each_day_weights = []
            for j in range(weights.shape[1]):
                each_day_weights.append(weights[i][j])
            data_to_weights.append((run_id, int(i + 1), str(each_day_weights)))

        data_to_details = [(run_id, "Boot_"+base, self.gamma2, self.aggregate_method, self.n_day_rebalance, self.stockdata.train_start, self.stockdata.train_end,
                            self.stockdata.test_end, cum_ret.iloc[-1], sharpe, vo, to, mdd, "NA", "NA", "NA", "NA",
                            "NA", "NA", len(self.stockdata.stocks_label), str(self.stockdata.stocks_label))]

        database.append_to_weights(data_to_weights)
        database.append_to_details(data_to_details)

        if print_status:
            print("Run details have been saved.")
            print(f"j = {j} || return: {cum_ret.iloc[-1]}, sharpe: {sharpe}, Volatility: {vo}, Turnover Rate: {to}, Maximum Drawdown: {mdd}")
    
    # #
    # """
    # r_l: historical return data for training, (n,t) matrix
    # """
    # @staticmethod
    # def __growth_optimal_portfolio(r_l):  # 20240308
    #     try:
    #         cov_mat = LedoitWolf().fit(r_l.T.to_numpy()).covariance_
    #         mu = r_l.T.mean().to_numpy()
    #     except:
    #         cov_mat = LedoitWolf().fit(r_l.T).covariance_
    #         mu = np.mean(r_l, axis=1)
    #
    #     # cov_mat = LedoitWolf().fit(r_l.T).covariance_  # changed cov method
    #     # mu = np.mean(r_l, axis=1)
    #     P = np.linalg.cholesky(cov_mat)
    #     G_factor = P
    #     G_factor_T = G_factor.T
    #
    #     with Model("growth_optimal_portfolio") as m:
    #         n, t = r_l.shape
    #         weights = m.variable(n, Domain.greaterThan(0.0))
    #         r = m.variable(1, Domain.unbounded())
    #         growth = Expr.sub(Expr.dot(mu, weights), r)
    #
    #         m.objective(ObjectiveSense.Maximize, growth)
    #         m.constraint('budget', Expr.sum(weights), Domain.equalsTo(1.0))
    #         m.constraint('variance', Expr.vstack(1, r, Expr.mul(G_factor_T, weights)), Domain.inRotatedQCone())
    #
    #         m.solve()
    #         w = weights.level()
    #
    #     return w
    #
    # def __kelly_metrics(self, print_status=True):
    #     X, X_train, X_test = self.stockdata.get_stock_data()
    #
    #     T_test, N_test = X_test.shape  # number of stocks, number of days
    #
    #     X_test = X_test.to_numpy()
    #
    #     rolling_weights = self.generic_rolling_window_helper("kelly", X, len(X_train), self.n_day_rebalance)
    #
    #     # weights = rolling_weights
    #     returns = 1+(X_test * rolling_weights).sum(axis=1)
    #
    #     # Calculate Sharpe Ratio
    #     sharpe = np.mean(returns-1) / np.std(returns-1) * np.sqrt(252)
    #
    #     # Calculate Volatility
    #     vo = np.sqrt(252) * np.std(returns)  # 20240311
    #
    #     # Calculate Turnover Rate
    #     # to = "No applicable"
    #     total = 0
    #     for k in range(T_test - 2):  # loop to M-2
    #         # Calculate difference between weights at k and weights at k+1
    #         total += TO(X_test[k]+1, rolling_weights[k], rolling_weights[k+1])  # modified
    #     to = total / (T_test - 1)  # M-R-1
    #
    #     # Calculate Maximum Drawdown
    #     cum_ret = returns.cumprod()
    #     cum_ret = pd.Series(cum_ret)
    #     window = 252
    #     Roll_Max = cum_ret.rolling(window, min_periods=1).max()
    #     Daily_Drawdown = cum_ret / Roll_Max - 1.0
    #     Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
    #     mdd = Max_Daily_Drawdown.min()*-1
    #     # For calculation
    #     cum_ret = cum_ret - 1
    #
    #     self.returns = returns
    #     self.sharpe = sharpe
    #     self.vo = vo
    #     self.to = to
    #     self.mdd = mdd
    #
    #     # Import to database
    #     run_id = "Run" + database.get_last_id()
    #     data_to_weights = []
    #     for i in range(len(rolling_weights)):
    #         each_day_weights = []
    #         for j in range(rolling_weights.shape[1]):
    #             each_day_weights.append(rolling_weights[i][j])
    #         data_to_weights.append((run_id, int(i + 1), str(each_day_weights)))
    #
    #     data_to_details = [(run_id, "Kelly", self.gamma2, self.aggregate_method, self.n_day_rebalance, self.stockdata.train_start, self.stockdata.train_end,
    #                         self.stockdata.test_end, cum_ret.iloc[-1], sharpe, vo, to, mdd, "NA", "NA", "NA", "NA",
    #                         "NA", "NA", len(self.stockdata.stocks_label), str(self.stockdata.stocks_label))]
    #
    #     database.append_to_weights(data_to_weights)
    #     database.append_to_details(data_to_details)
    #
    #     if print_status:
    #         print("Run details have been saved.")
    #         print(f"return: {cum_ret.iloc[-1]}, sharpe: {sharpe}, Volatility: {vo}, Turnover Rate: {to}, Maximum Drawdown: {mdd}")


    # @staticmethod
    # def __growth_optimal_portfolio_cone(r_l):  # 20240317
    #     with Model("growth_optimal_portfolio_cone") as m:
    #         n, t = r_l.shape
    #         weights = m.variable(n, Domain.greaterThan(0.0))
    #         growth = Expr.mul(weights, r_l)
    #         T = m.variable(t, Domain.unbounded())
    #
    #         m.objective(ObjectiveSense.Maximize, Expr.sum(T))
    #         m.constraint('budget', Expr.sum(weights), Domain.equalsTo(1.0))
    #         m.constraint('exp', Expr.hstack(growth, Expr.ones(t), T), Domain.inPExpCone())
    #         # m.constraint('long_only',weights, Domain.greaterThan(0.0))
    #
    #         m.solve()
    #
    #         # Check if the solution is an optimal point
    #         solsta = m.getPrimalSolutionStatus()
    #         if (solsta != SolutionStatus.Optimal):
    #             # See https://docs.mosek.com/latest/pythonfusion/accessing-solution.html about handling solution statuses.
    #             raise Exception(f"Unexpected solution status: {solsta}")
    #
    #         w = weights.level()
    #
    #     return w
    #
    # # Variables & parameters
    # """
    # t:   number of periods of return data
    # n:   number of assets
    # r_l: historical return data for training, (n,t) matrix
    # n_1: number of resamples
    # n_2: size of each resample
    # n_3: number of resampled subsets
    # n_4: size of each subset
    # """
    # def __ego(self, r_l: '(n,t) matrix',  # 20240317
    #                 n_1: 'number of resamples',
    #                 n_2: 'size of each resample',
    #                 n_3: 'number of resampled subsets',
    #                 n_4: 'size of each subset',
    #                 cone: 'cone or not'):
    #     # sample mean and covariance matrix to generate returns for resampleing
    #     cov_mat = LedoitWolf().fit(np.cov(r_l)).covariance_  # changed cov method
    #     mean=np.mean(r_l, axis=1)
    #
    #     if len(mean) != cov_mat.shape[0]:
    #         print(f'size mismatch:\n expected return len: {len(mean)} \n size of cov_mat: {cov_mat.shape}')
    #
    #     # store number of assets as n
    #     n, t = r_l.shape
    #     # matrix to store results from each resample iterations, n_1 sets of weights in total
    #     EOGP_w = np.empty((n_1, n))
    #
    #     # resample n_1 times
    #     # for h in tqdm(range(n_1)):
    #     for h in range(n_1):
    #         # create n_2 days of returns for subset portfolios
    #         sample_day_index=np.random.choice(t, n_2)
    #         resampled_return = r_l.T[sample_day_index].T
    #
    #         # resampled_return = np.random.default_rng().multivariate_normal(mean, cov_mat, size=n_2).T
    #
    #         # matrix to store results from each subset, n_3 sets of weights for each resample iteration
    #         resample_w = np.empty((n_3, n))
    #
    #         # create n_3 subset portfolios
    #         for j in range(n_3):
    #
    #             # create subset of n_4 stocks without replacement
    #             subset = np.random.choice(n, size=n_4, replace=False)
    #             subset_r_l = np.array([resampled_return[index] for index in subset])
    #
    #             # fit optimal growth portfolio for the subset
    #             if not cone:
    #                 subset_w = self.__growth_optimal_portfolio(subset_r_l)
    #             else:
    #                 subset_w = self.__growth_optimal_portfolio_cone(subset_r_l)
    #
    #             # array
    #             set_w = np.zeros(n)
    #             # assign corresponding weights to each stock; for stock not in the subset, weight remains as 0
    #             for count, index in enumerate(subset):
    #                 set_w[index] = subset_w[count]
    #
    #             # store results in matrix
    #             resample_w[j] = set_w
    #
    #         # Aggregate weights
    #         if self.aggregate_method == "sample_mean":
    #             aggregated_resample_w = np.mean(resample_w, axis=0)
    #         else:
    #             aggregated_resample_w = compute_geometric_median(resample_w, weights=None).median
    #         # store results in matrix
    #         EOGP_w[h] = aggregated_resample_w
    #
    #     # calculate final results after all resampling by taking mean of weights for each asset from each resampling
    #     if self.aggregate_method == "sample_mean":
    #         aggregated_EOGP_w = np.mean(EOGP_w, axis=0)
    #     else:
    #         aggregated_EOGP_w = compute_geometric_median(EOGP_w, weights=None).median
    #
    #     return aggregated_EOGP_w

    # Bootstrap_SSR Portfolio
    """
    Parameters:
    r_l: historical return data for training, (n,t) matrix
    n_1: number of resamples
    n_2: size of each resample
    n_3: number of resampled subsets
    n_4: size of each subset
    """
    def __boot_ssr_mvp(self, r_l: '(n,t) matrix',
                       n_1: 'number of resamples',
                       n_2: 'size of each resample',
                       n_3: 'number of resampled subsets',
                       n_4: 'size of each subset'):

        # Store number of assets as n
        n, t = r_l.shape

        # Matrix to store results from each resample iterations, n_1 sets of weights in total
        BS_SSR_MVP_w = np.empty((n_1, n))

        # Resample n_1 times
        for h in range(n_1):
            # Create n_2 days of returns for subset portfolios
            sample_day_index=np.random.choice(t, n_2)
            resampled_return = r_l.T[sample_day_index].T

            # Matrix to store results from each subset, n_3 sets of weights for each resample iteration
            resample_w = np.empty((n_3, n))

            # Create n_3 subset portfolios
            for j in range(n_3):

                # Create subset of n_4 stocks without replacement
                subset = np.random.choice(n, size=n_4, replace=False)
                subset_r_l = np.array([resampled_return[index] for index in subset])

                # Fit optimal growth portfolio for the subset
                subset_w = self.__mvp(subset_r_l.T)

                # Array
                set_w = np.zeros(n)

                # Assign corresponding weights to each stock; for stock not in the subset, weight remains as 0
                for count, index in enumerate(subset):
                    set_w[index] = subset_w[count]

                # Store results in matrix
                resample_w[j] = set_w

            # Calculate resulting weights for this iteration of resample by taking mean of weights
            # for each asset from each subset portfolio
            if self.aggregate_method == "sample_mean":
                aggregated_resample_w = np.mean(resample_w, axis=0)
            else:
                aggregated_resample_w = compute_geometric_median(resample_w, weights=None).median
            # Store results in matrix
            BS_SSR_MVP_w[h] = aggregated_resample_w

        # Calculate final results after all resampling by taking mean of weights for each asset from each resampling
        if self.aggregate_method == "sample_mean":
            aggregated_BS_SSR_MVP_w = np.mean(BS_SSR_MVP_w, axis=0)
        else:
            aggregated_BS_SSR_MVP_w = compute_geometric_median(BS_SSR_MVP_w, weights=None).median

        return aggregated_BS_SSR_MVP_w

    # SSR_Bootstrap Portfolio
    """
    Parameters:
    r_l: historical return data for training, (n,t) matrix
    n_1: number of resamples
    n_2: size of each resample
    n_3: number of resampled subsets
    n_4: size of each subset
    """
    def __ssr_boot_mvp(self, r_l: '(n,t) matrix',
                    n_1: 'number of resamples',
                    n_2: 'size of each resample',
                    n_3: 'number of resampled subsets',
                    n_4: 'size of each subset'):
        # Store number of assets as n
        n, t = r_l.shape

        # Matrix to store results from each resample iterations, n_1 sets of weights in total
        SSR_BS_MVP_w = np.empty((n_3, n))

        # Resample n_1 times
        for h in range(n_3):
            # Create subset of n_4 stocks without replacement
            subset = np.random.choice(n, size=n_4, replace=False)
            subset_r_l = np.array([r_l[index] for index in subset])

            # Matrix to store results from each subset, n_3 sets of weights for each resample iteration
            subset_w = np.empty((n_1, n))

            # Create n_3 subset portfolios
            for j in range(n_1):

                # Create n_2 days of returns for subset portfolios
                sample_day_index=np.random.choice(t, n_2)
                resampled_subset_return = subset_r_l.T[sample_day_index].T

                # Fit optimal growth portfolio for the subset
                resampled_subset_w = self.__mvp(resampled_subset_return.T)

                # Array
                set_w = np.zeros(n)

                # Assign corresponding weights to each stock; for stock not in the subset, weight remains as 0
                for count, index in enumerate(subset):
                    set_w[index] = resampled_subset_w[count]

                # Store results in matrix
                subset_w[j] = set_w

            # Calculate resulting weights for this iteration of resample by taking mean of weights
            # for each asset from each subset portfolio
            if self.aggregate_method == "sample_mean":
                aggregated_resample_w = np.mean(subset_w, axis=0)
            else:
                aggregated_resample_w = compute_geometric_median(subset_w, weights=None).median
            # Store results in matrix
            SSR_BS_MVP_w[h] = aggregated_resample_w

        # Calculate final results after all resampling by taking mean of weights for each asset from each resampling
        if self.aggregate_method == "sample_mean":
            aggregated_SSR_BS_MVP_w = np.mean(SSR_BS_MVP_w, axis=0)
        else:
            aggregated_SSR_BS_MVP_w = compute_geometric_median(SSR_BS_MVP_w, weights=None).median

        return aggregated_SSR_BS_MVP_w

    # # Variables & parameters
    # """
    # X: historical return data including test period
    # n_1: number of resamples
    # n_2: size of each resample
    # n_3: number of resampled subsets
    # n_4: size of each subset
    # """
    # def __ego_metrics(self, n_1, n_2, n_3, n_4, cone=False, print_status=True):
    #     X, X_train, X_test = self.stockdata.get_stock_data()
    #
    #     T_test, N_test = X_test.shape  # number of stocks, number of days
    #
    #     X_test = X_test.to_numpy()
    #
    #     if not cone:  # modified
    #         rolling_weights = self.generic_rolling_window_helper("ego", X, len(X_train), self.n_day_rebalance)
    #     else:  # modified
    #         rolling_weights = self.generic_rolling_window_helper("ego_cone", X, len(X_train), self.n_day_rebalance)
    #
    #     # weights = rolling_weights
    #     returns = 1+(X_test * rolling_weights).sum(axis=1)
    #
    #     # Calculate Sharpe Ratio
    #     sharpe = np.mean(returns-1) / np.std(returns-1) * np.sqrt(252)
    #
    #     # Calculate Volatility
    #     vo = np.sqrt(252) * np.std(returns)  # 20240311
    #
    #     # Calculate Turnover Rate
    #     # to = "No applicable"
    #     total = 0
    #     for k in range(T_test - 2):  # loop to M-2
    #         # Calculate difference between weights at k and weights at k+1
    #         total += TO(X_test[k]+1, rolling_weights[k], rolling_weights[k+1])  # modified
    #     to = total / (T_test - 1)  # M-R-1
    #
    #     # Calculate Maximum Drawdown
    #     cum_ret = returns.cumprod()
    #     cum_ret = pd.Series(cum_ret)
    #     window = 252
    #     Roll_Max = cum_ret.rolling(window, min_periods=1).max()
    #     Daily_Drawdown = cum_ret / Roll_Max - 1.0
    #     Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
    #     mdd = Max_Daily_Drawdown.min()*-1
    #     # For calculation
    #     cum_ret = cum_ret - 1
    #
    #     self.returns = returns
    #     self.sharpe = sharpe
    #     self.vo = vo
    #     self.to = to
    #     self.mdd = mdd
    #
    #     # Import to database
    #     run_id = "Run" + database.get_last_id()
    #     data_to_weights = []
    #     for i in range(len(rolling_weights)):
    #         each_day_weights = []
    #         for j in range(rolling_weights.shape[1]):
    #             each_day_weights.append(rolling_weights[i][j])
    #         data_to_weights.append((run_id, int(i + 1), str(each_day_weights)))
    #
    #     extra = ""
    #     if cone:
    #         extra = "_cone"
    #
    #     data_to_details = [(run_id, "EGO"+extra, self.gamma2, self.aggregate_method, self.n_day_rebalance, self.stockdata.train_start, self.stockdata.train_end,
    #                         self.stockdata.test_end, cum_ret.iloc[-1], sharpe, vo, to, mdd, "NA", "NA", n_1, n_2,
    #                         n_3, n_4, len(self.stockdata.stocks_label), str(self.stockdata.stocks_label))]
    #
    #     database.append_to_weights(data_to_weights)
    #     database.append_to_details(data_to_details)
    #
    #     if print_status:
    #         print("Run details have been saved.")
    #         print(f"Number of Resamples = {n_1}, Size of Each Resample = {n_2}, Number of Resampled Subsets = {n_3}, Size of Each Subset = {n_4} || return: {cum_ret.iloc[-1]}, sharpe: {sharpe}, Volatility: {vo}, Turnover Rate: {to}, Maximum Drawdown: {mdd}")

    def __boot_ssr_mvp_metrics(self, n_1, n_2, n_3, n_4, print_status=True):
        X, X_train, X_test = self.stockdata.get_stock_data()
        T_test, N_test = X_test.shape  # number of stocks, number of days
        X_test = X_test.to_numpy()

        rolling_weights = self.generic_rolling_window_helper("boot_ssr_mvp", X, len(X_train), self.n_day_rebalance)
        returns = 1+(X_test * rolling_weights).sum(axis=1)

        # Calculate Sharpe Ratio
        sharpe = np.mean(returns-1) / np.std(returns-1) * np.sqrt(252)

        # Calculate Volatility
        vo = np.sqrt(252) * np.std(returns)  # 20240311

        # Calculate Turnover Rate
        total = 0
        for k in range(T_test - 2):  # loop to M-2
            # Calculate difference between weights at k and weights at k+1
            total += TO(X_test[k]+1, rolling_weights[k], rolling_weights[k+1])  # modified
        to = total / (T_test - 1)  # M-R-1

        # Calculate Maximum Drawdown
        cum_ret = returns.cumprod()
        cum_ret = pd.Series(cum_ret)
        window = 252
        Roll_Max = cum_ret.rolling(window, min_periods=1).max()
        Daily_Drawdown = cum_ret / Roll_Max - 1.0
        Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
        mdd = Max_Daily_Drawdown.min()*-1

        # For calculation
        cum_ret = cum_ret - 1

        self.returns = returns
        self.sharpe = sharpe
        self.vo = vo
        self.to = to
        self.mdd = mdd

        # Import to database
        run_id = "Run" + database.get_last_id()
        data_to_weights = []
        for i in range(len(rolling_weights)):
            each_day_weights = []
            for j in range(rolling_weights.shape[1]):
                each_day_weights.append(rolling_weights[i][j])
            data_to_weights.append((run_id, int(i + 1), str(each_day_weights)))

        data_to_details = [(run_id, "BOOT_SSR_MVP", self.gamma2, self.aggregate_method, self.n_day_rebalance, self.stockdata.train_start, self.stockdata.train_end,
                            self.stockdata.test_end, cum_ret.iloc[-1], sharpe, vo, to, mdd, "NA", "NA", n_1, n_2,
                            n_3, n_4, len(self.stockdata.stocks_label), str(self.stockdata.stocks_label))]

        database.append_to_weights(data_to_weights)
        database.append_to_details(data_to_details)

        if print_status:
            print("Run details have been saved.")
            print(f"Number of Resamples = {n_1}, Size of Each Resample = {n_2}, Number of Resampled Subsets = {n_3}, Size of Each Subset = {n_4} || return: {cum_ret.iloc[-1]}, sharpe: {sharpe}, Volatility: {vo}, Turnover Rate: {to}, Maximum Drawdown: {mdd}")

    def __ssr_boot_mvp_metrics(self, n_1, n_2, n_3, n_4, print_status=True):
        X, X_train, X_test = self.stockdata.get_stock_data()
        T_test, N_test = X_test.shape  # number of stocks, number of days
        X_test = X_test.to_numpy()

        rolling_weights = self.generic_rolling_window_helper("ssr_boot_mvp", X, len(X_train), self.n_day_rebalance)
        returns = 1+(X_test * rolling_weights).sum(axis=1)

        # Calculate Sharpe Ratio
        sharpe = np.mean(returns-1) / np.std(returns-1) * np.sqrt(252)

        # Calculate Volatility
        vo = np.sqrt(252) * np.std(returns)  # 20240311

        # Calculate Turnover Rate
        total = 0
        for k in range(T_test - 2):  # loop to M-2
            # Calculate difference between weights at k and weights at k+1
            total += TO(X_test[k]+1, rolling_weights[k], rolling_weights[k+1])  # modified
        to = total / (T_test - 1)  # M-R-1

        # Calculate Maximum Drawdown
        cum_ret = returns.cumprod()
        cum_ret = pd.Series(cum_ret)
        window = 252
        Roll_Max = cum_ret.rolling(window, min_periods=1).max()
        Daily_Drawdown = cum_ret / Roll_Max - 1.0
        Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
        mdd = Max_Daily_Drawdown.min()*-1

        # For calculation
        cum_ret = cum_ret - 1

        self.returns = returns
        self.sharpe = sharpe
        self.vo = vo
        self.to = to
        self.mdd = mdd

        # Import to database
        run_id = "Run" + database.get_last_id()
        data_to_weights = []
        for i in range(len(rolling_weights)):
            each_day_weights = []
            for j in range(rolling_weights.shape[1]):
                each_day_weights.append(rolling_weights[i][j])
            data_to_weights.append((run_id, int(i + 1), str(each_day_weights)))

        data_to_details = [(run_id, "SSR_BOOT_MVP", self.gamma2, self.aggregate_method, self.n_day_rebalance, self.stockdata.train_start, self.stockdata.train_end,
                            self.stockdata.test_end, cum_ret.iloc[-1], sharpe, vo, to, mdd, "NA", "NA", n_1, n_2,
                            n_3, n_4, len(self.stockdata.stocks_label), str(self.stockdata.stocks_label))]

        database.append_to_weights(data_to_weights)
        database.append_to_details(data_to_details)

        if print_status:
            print("Run details have been saved.")
            print(f"Number of Resamples = {n_1}, Size of Each Resample = {n_2}, Number of Resampled Subsets = {n_3}, Size of Each Subset = {n_4} || return: {cum_ret.iloc[-1]}, sharpe: {sharpe}, Volatility: {vo}, Turnover Rate: {to}, Maximum Drawdown: {mdd}")

    def set_disable_tqdm(self, disable):
        if disable == True or disable == False:
            self.disable_tqdm = disable
        else:
            print("Please input True or False")
            return None

    def set_n_day_rebalance(self, n_day_rebalance):
        self.n_day_rebalance = n_day_rebalance

    def set_ssr_properties(self, s, b):
        self.ssr_properties = [s, b]

    def set_ensemble_properties(self, n_1, n_2, n_3, n_4):
        self.ensemble_properties = [n_1, n_2, n_3, n_4]

    def methods_metrics(self, method, print_status=True):
        if method == "ewp":
            return self.__ewp(print_status)
        elif method == "mvp":
            return self.__mvp_metrics(print_status)
        elif method == "ssr_mvp":
            if self.ssr_properties[0] == 0 or self.ssr_properties[1] == 0:
                print("Please set ssr properties.")
                return None
            return self.__ssr_metrics(self.ssr_properties[0], self.ssr_properties[1], "mvp", print_status)
        elif method == "boot_ssr_mvp" or method == "ssr_boot_mvp":
            res = True in (ele == 0 for ele in self.ensemble_properties)
            if res:
                print("Please set ego properties.")
                return None
            elif method == "boot_ssr_mvp":
                return self.__boot_ssr_mvp_metrics(self.ensemble_properties[0], self.ensemble_properties[1],
                                                   self.ensemble_properties[2], self.ensemble_properties[3],
                                                   print_status)
            else:
                return self.__ssr_boot_mvp_metrics(self.ensemble_properties[0], self.ensemble_properties[1],
                                                   self.ensemble_properties[2], self.ensemble_properties[3],
                                                   print_status)
        elif method == "boot_mvp":
            return self.__boot_metrics("mvp", print_status)
        else:
            print("Please input a correct method (ewp/mvp/ssr_mvp/boot_mvp/ssr_boot_mvp/boot_ssr_mvp).")

    def plot_histogram(self, method, size, print_status=True):
        returns_arr, sharpe_arr, vo_arr, to_arr, mdd_arr = [], [], [], [], []

        for i in tqdm(range(size)):
            if method == "ewp":
                self.methods_metrics("ewp", print_status)
            elif method == "mvp":
                self.methods_metrics("mvp", print_status)
            elif method == "ssr_mvp":
                self.methods_metrics("ssr_mvp", print_status)
            elif method == "boot_mvp":
                self.methods_metrics("boot_mvp", print_status)
            elif method == "ssr_boot_mvp":
                self.methods_metrics("ssr_boot_mvp", print_status)
            elif method == "boot_ssr_mvp":
                self.methods_metrics("boot_ssr_mvp", print_status)

            returns_arr.append(self.returns.cumprod()[-1]-1)
            sharpe_arr.append(self.sharpe)
            vo_arr.append(self.vo)
            to_arr.append(self.to)
            mdd_arr.append(self.mdd)

        # Plot returns
        bin_size = size
        if int(bin_size/4) > 20:
            bin_size = 20
        else:
            bin_size = int(bin_size/4)

        # Plotting a basic histogram
        plt.hist(returns_arr, bins=bin_size, color='skyblue', edgecolor='black')

        # Adding labels and title
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.title('Returns in ' + method)
        mean = 'Mean:' + str(np.mean(returns_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')

        # Display the plot
        plt.show()

        plt.hist(sharpe_arr, bins=bin_size, color='skyblue', edgecolor='black')
        plt.xlabel('Sharpe ratio')
        plt.ylabel('Frequency')
        plt.title('Sharpe ratio in ' + method)
        mean = 'Mean:' + str(np.mean(sharpe_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')
        plt.show()

        plt.hist(vo_arr, bins=bin_size, color='skyblue', edgecolor='black')
        plt.xlabel('Volatility')
        plt.ylabel('Frequency')
        plt.title('Volatility in ' + method)
        mean = 'Mean:' + str(np.mean(vo_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')
        plt.show()

        plt.hist(to_arr, bins=bin_size, color='skyblue', edgecolor='black')
        plt.xlabel('Turnover rate')
        plt.ylabel('Frequency')
        plt.title('Turnover rate in ' + method)
        mean = 'Mean:' + str(np.mean(to_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')
        plt.show()

        plt.hist(mdd_arr, bins=bin_size, color='skyblue', edgecolor='black')
        plt.xlabel('Maximum drawdown')
        plt.ylabel('Frequency')
        plt.title('Maximum drawdown in ' + method)
        mean = 'Mean:' + str(np.mean(mdd_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')
        plt.show()

    def plot_random_time(self, method, size, train_period, test_period, print_status=True):    
        counter = 0
        train_start_arr, train_end_arr, test_end_arr = [], [], []
        returns_arr, sharpe_arr, vo_arr, to_arr, mdd_arr = [], [], [], [], []

        while counter < size:
            train_start = random_date(train_period, test_period)
            train_end = train_start + timedelta(train_period)
            test_end = train_end + timedelta(test_period)
            if train_start < train_end:
                train_start_arr.append(train_start.strftime("%Y-%m-%d"))
                train_end_arr.append(train_end.strftime("%Y-%m-%d"))
                test_end_arr.append(test_end.strftime("%Y-%m-%d"))
                counter += 1

        for i in tqdm(range(size)):
            self.stockdata.train_start = train_start_arr[i]
            self.stockdata.train_end = train_end_arr[i]
            self.stockdata.test_end = test_end_arr[i]
            if method == "ewp":
                self.methods_metrics("ewp", print_status)
            elif method == "mvp":
                self.methods_metrics("mvp", print_status)
            elif method == "ssr_mvp":
                self.methods_metrics("ssr_mvp", print_status)
            elif method == "boot_mvp":
                self.methods_metrics("boot_mvp", print_status)
            elif method == "ssr_boot_mvp":
                self.methods_metrics("ssr_boot_mvp", print_status)
            elif method == "boot_ssr_mvp":
                self.methods_metrics("boot_ssr_mvp", print_status)
            returns_arr.append(self.returns.cumprod()[-1]-1)
            sharpe_arr.append(self.sharpe)
            vo_arr.append(self.vo)
            to_arr.append(self.to)
            mdd_arr.append(self.mdd)

        # Plot returns
        bin_size = size
        if int(bin_size / 4) > 20:
            bin_size = 20
        else:
            bin_size = int(bin_size / 4)

        # Plotting a basic histogram
        plt.hist(returns_arr, bins=bin_size, color='skyblue', edgecolor='black')

        # Adding labels and title
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.title('Returns in ' + method)
        mean = 'Mean:' + str(np.mean(returns_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')

        # Display the plot
        plt.show()

        plt.hist(sharpe_arr, bins=bin_size, color='skyblue', edgecolor='black')
        plt.xlabel('Sharpe ratio')
        plt.ylabel('Frequency')
        plt.title('Sharpe ratio in ' + method)
        mean = 'Mean:' + str(np.mean(sharpe_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')
        plt.show()

        plt.hist(vo_arr, bins=bin_size, color='skyblue', edgecolor='black')
        plt.xlabel('Volatility')
        plt.ylabel('Frequency')
        plt.title('Volatility in ' + method)
        mean = 'Mean:' + str(np.mean(vo_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')
        plt.show()

        plt.hist(to_arr, bins=bin_size, color='skyblue', edgecolor='black')
        plt.xlabel('Turnover rate')
        plt.ylabel('Frequency')
        plt.title('Turnover rate in ' + method)
        mean = 'Mean:' + str(np.mean(to_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')
        plt.show()

        plt.hist(mdd_arr, bins=bin_size, color='skyblue', edgecolor='black')
        plt.xlabel('Maximum drawdown')
        plt.ylabel('Frequency')
        plt.title('Maximum drawdown in ' + method)
        mean = 'Mean:' + str(np.mean(mdd_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')
        plt.show()

    def plot_random_time_random_stock(self, method: str, size: int, train_period: int, test_period: int,
                                      stock_list_size: int, fix_stock_time=None, print_status: bool=True):
        self.stockdata.test_stock_list = None
        clean_X = self.stockdata.get_stock_data()[2]  # Do not remove
        clean_X = None  # Do not remove
        all_stocks = self.stockdata.stocks_label

        counter = 0
        train_start_arr, train_end_arr, test_end_arr = [], [], []
        stock_list_arr=[]
        returns_arr, sharpe_arr, vo_arr, to_arr, mdd_arr = [], [], [], [], []

        # Fix backtesting with pickle
        if fix_stock_time:
            with open(fix_stock_time, 'rb') as f:
                train_start_arr, train_end_arr, test_end_arr, stock_list_arr = pickle.load(f)

        else:
            # Random periods and stocks
            while counter < size:
                train_start = random_date(train_period, test_period)
                train_end = train_start + timedelta(train_period)
                test_end = train_end + timedelta(test_period)
                stock_list = random.sample(self.stockdata.stocks_label[:-1], stock_list_size)+['Cash']
                if train_start < train_end:
                    train_start_arr.append(train_start.strftime("%Y-%m-%d"))
                    train_end_arr.append(train_end.strftime("%Y-%m-%d"))
                    test_end_arr.append(test_end.strftime("%Y-%m-%d"))
                    stock_list_arr.append(stock_list)
                    counter += 1

            time = str(datetime.now()).replace(':','_').replace('.','_').replace(' ','_')

            # Stores periods and stocks as pickle
            with open(f'stock_time_{time}.pickle', 'wb') as f:
                pickle.dump((train_start_arr, train_end_arr, test_end_arr, stock_list_arr), f)

        # In case if crash at nth iteration; restart at n+1
        # train_start_arr=train_start_arr[n:]
        # train_end_arr=train_end_arr[n:]
        # test_end_arr=test_end_arr[n:]
        # stock_list_arr=stock_list_arr[n:]

        [stock_list.remove("Cash") for stock_list in stock_list_arr]

        for i in tqdm(range(size)):
            self.stockdata.train_start = train_start_arr[i]
            self.stockdata.train_end = train_end_arr[i]
            self.stockdata.test_end = test_end_arr[i]
            self.stockdata.test_stock_list=stock_list_arr[i]
            if method == "ewp":
                self.methods_metrics("ewp", print_status)
            elif method == "mvp":
                self.methods_metrics("mvp", print_status)
            elif method == "ssr_mvp":
                self.methods_metrics("ssr_mvp", print_status)
            elif method == "boot_mvp":
                self.methods_metrics("boot_mvp", print_status)
            elif method == "ssr_boot_mvp":
                self.methods_metrics("ssr_boot_mvp", print_status)
            elif method == "boot_ssr_mvp":
                self.methods_metrics("boot_ssr_mvp", print_status)
            returns_arr.append(self.returns.cumprod()[-1]-1)
            sharpe_arr.append(self.sharpe)
            vo_arr.append(self.vo)
            to_arr.append(self.to)
            mdd_arr.append(self.mdd)

        # Reset stock labels in stockdata
        self.stockdata.test_stock_list = None
        self.stockdata.stocks_label = all_stocks

        # Plot returns
        bin_size = size
        if int(bin_size / 4) > 20:
            bin_size = 20
        else:
            bin_size = int(bin_size / 4)

        # Plotting a basic histogram
        plt.hist(returns_arr, bins=bin_size, color='skyblue', edgecolor='black')

        # Adding labels and title
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.title('Returns in ' + method)
        mean = 'Mean:' + str(np.mean(returns_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')

        # Display the plot
        plt.show()

        plt.hist(sharpe_arr, bins=bin_size, color='skyblue', edgecolor='black')
        plt.xlabel('Sharpe ratio')
        plt.ylabel('Frequency')
        plt.title('Sharpe ratio in ' + method)
        mean = 'Mean:' + str(np.mean(sharpe_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')
        plt.show()

        plt.hist(vo_arr, bins=bin_size, color='skyblue', edgecolor='black')
        plt.xlabel('Volatility')
        plt.ylabel('Frequency')
        plt.title('Volatility in ' + method)
        mean = 'Mean:' + str(np.mean(vo_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')
        plt.show()

        plt.hist(to_arr, bins=bin_size, color='skyblue', edgecolor='black')
        plt.xlabel('Turnover rate')
        plt.ylabel('Frequency')
        plt.title('Turnover rate in ' + method)
        mean = 'Mean:' + str(np.mean(to_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')
        plt.show()

        plt.hist(mdd_arr, bins=bin_size, color='skyblue', edgecolor='black')
        plt.xlabel('Maximum drawdown')
        plt.ylabel('Frequency')
        plt.title('Maximum drawdown in ' + method)
        mean = 'Mean:' + str(np.mean(mdd_arr))
        plt.annotate(mean, xy=(0.03, 0.95), xycoords='axes fraction')
        plt.show()

    def __generic_rolling_window(self,
                                 strategy,
                                 r: '(n,T) matrix',
                                 t: 'number of train days',
                                 n_day_rebalance: 'number of days between rebalance'):
        n = r.shape[0]

        # Create daily window views of shape (n,t)
        view = window_view(r, t)[:-1]

        # Identify indexes of windows that require portfolio construction
        window_count = view.shape[0]
        portfolio_construction_index = np.arange(0, window_count, n_day_rebalance)

        # Array to store the results **shape=(# of portfolios, size of each portfolio, which is n)
        results = np.zeros((len(portfolio_construction_index), n))

        # Select portfolio
        for count, window_index in tqdm(enumerate(portfolio_construction_index), disable=self.disable_tqdm):
            r_l = view[window_index]
            if strategy == "ewp":
                portfolio = [1/n] * n
            elif strategy == "mvp":
                r_l = pd.DataFrame(data=r_l)
                portfolio = self.__mvp(r_l.T+1)  # (T,n)
            elif strategy == "ssr_mvp":
                r_l = r_l.T
                r_l = pd.DataFrame(data=r_l, columns=self.stockdata.stocks_label)
                portfolio = self.__ssr(r_l, self.ssr_properties[0], self.ssr_properties[1], "mvp")  # (T,n)
            elif strategy == "ssr_boot_mvp":
                portfolio = self.__ssr_boot_mvp(r_l + 1, self.ensemble_properties[0], self.ensemble_properties[1],  # (n,T)
                                                self.ensemble_properties[2], self.ensemble_properties[3])
            elif strategy == "boot_ssr_mvp":
                portfolio = self.__boot_ssr_mvp(r_l + 1, self.ensemble_properties[0], self.ensemble_properties[1],  # (n,T)
                                                self.ensemble_properties[2], self.ensemble_properties[3])

            results[count] = portfolio

        return results  # , view, portfolio_construction_index

    def generic_rolling_window_helper(self,
                                      strategy,
                                      r: '(T,n) matrix',
                                      t: 'number of days per window',
                                      n_day_rebalance: 'number of days between rebalance',):
        r = r.T
        portfolios = self.__generic_rolling_window(strategy, r, t, n_day_rebalance)

        if n_day_rebalance == 1:
            return portfolios

        backtest_days = r.to_numpy()[:, t:].copy().T
        t_days, n_stocks = backtest_days.shape

        daily_portfolio = np.empty((t_days, n_stocks))

        for i in range(t_days):
            if i % n_day_rebalance == 0:
                index = int(i / n_day_rebalance)
                daily_portfolio[i] = portfolios[index]

            else:
                next_day_portfolio = portfolio_after_return(daily_portfolio[i - 1], backtest_days[i - 1]+1)  # modified
                daily_portfolio[i] = next_day_portfolio

        return daily_portfolio


# Helper functions
def random_date(train_period=365, test_period=123):
    delta = datetime(2024, 3, 25) - datetime(2010, 1, 1)
    int_delta = delta.days
    random_days = randrange(int_delta - train_period - test_period)
    result = datetime(2010, 1, 1) + timedelta(random_days)
    return result


def window_view(data: '(n,T) matrix',
                t: 'number of days per window'):
    n = data.shape[0]
    shape = (n, t)
    view = np.lib.stride_tricks.sliding_window_view(data, shape)
    return view[0]


def portfolio_after_return(p, r):
    new_p = p*r / (p@r)
    return new_p


def TO(prev_return, prev_weight, new_weight):
    post_return_weight = portfolio_after_return(prev_weight, prev_return)
    TO = np.abs((new_weight - post_return_weight)).sum()
    return TO

