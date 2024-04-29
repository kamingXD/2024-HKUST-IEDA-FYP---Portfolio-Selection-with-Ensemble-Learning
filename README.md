# Portoflio Selection With Ensemble Learning

This project includes 3 classes for computing, back-testing and result backup.

* StockData
* GenericFunction
* Database

# StockData
* Variables

| Variables |  Usage   | Initial value         |
|--------|--------|-----------------------|
| `start_time` | Define stock data range fetching from Yahoo Finance | String: `2000-01-01`  |
| `end_time`  | Define stock data range fetching from Yahoo Finance   | String: `Today's date` |
| `train_start` | Train start date   | `None`|
| `train_end` | Train end date   | `None`|
| `test_end` | Test end date   | `None`|
| `test_stock_list` | Cache random stocklist for random_time_random_stock testing   | `None`|
| `alltime_stock_data` | Cache all unprocessed stock data in dataframe   | all data fetched from Yahoo Finance |
| `stocks_label` | Cache all stock labels in a list   | all stock labels in alltime_stock_data |

* Methods

| Method        |     Usage     |  Parameters  |
| ----------- |      ---------------     |  ------------------  |
| __get_all_time_stock_data(`self`) | Get data from YF and save to `self`  | NA |
| __get_stocks_label(`self`)  | Get stock labels and save to `self`   | NA |
| get_stock_data(`self`) | Return X, X_train and X_test according to `train_start`, `train_end` and `test_end`   | NA |

# GenericFunction

* Variables

| Variables        |     Usage     |  Initial value  |
| ----------- |      ---------------     |  ------------------  |
| `stockdata`  | Stores StockData object  | Pass by init |
| `n_day_rebalance`  | Stores rebalance period  | Int: `1` |
| `ssr_properties`  | Stores subset resampling's properties   | List<Int>: `[0,0]` |
| `ego_properties` | Stores ensemble's properties   | List<Int>: `[0,0,0,0]`|
| `boot_properties` | Stores bootstraping's properties   | Int: `50`|
| `gamma2`  | Gamma^2 value used in MVP | Int: `0.001` |
| `aggregate_method`  | Aggregation method (simple mean / geometric median) | String: `sample_mean` |
| `returns` | Stores return in different methods  | `None`|
| `sharpe` | Stores sharpe ratio in different methods  | `None`|
| `vo` | Stores volatility in different methods   | `None` |
| `to` | Stores turn over rate in different methods | `None` |
| `mdd` | Stores maximum drawdown in different methods | `None` |
| `disable_tqdm` | Disable tqdm (Progress bar) or not | `False` |

* Key methods

| Method        |     Usage     |  Parameters  |
| ----------- |      ---------------     |  ------------------  |
| methods_metrics(`self`, `method`, `print_status=True`)  | Execute methods, including `EWP`, `MVP`, `SSR`, `Bootstrapping`, `Ensemble` (SSR_BOOT_MVP/BOOT_SSR_MVP) | String: `method`, Boolean: `print_status` |
| plot_histogram(`self`, `method`, `metric`, `size`, `print_status=True`)  | Plot histogram of performance metrics with the specified histogram size and portfolio method listed above  | String: `method`, String: `metric`, Integer: `size`, Boolean: `print_status` |
| plot_random_time(`self`, `method`, `metric`, `size`, `test_period`, `print_status=True`)  | Plot histogram of performance metrics with the specified portfolio, histogram size, fixed test length with a randomized period  | String: `method`, String: `metric`, Integer: `size`, Integer: `test_period`, Boolean: `print_status` |
| plot_random_time_random_stock(`self`, `method`, `size`, `train_period`, `test_period`, `stock_list_size`, `fix_stock_time=None`, `print_status=True`)  | Plot histogram of performance metrics with the specified portfolio, histogram size, fixed test length with randomized stocks and period (Pass in a pickle for fixed sample)  | String: `method`, String: `metric`, Integer: `size`, Integer: `test_period`, Pickle: `fix_stock_time`, Boolean: `print_status` |

# Database

* Methods

| Method        |     Usage     |  Parameters  |
| ----------- |      ---------------     |  ------------------  |
| reset_table(`table_name`) | Reset database's table  | String: `table_name` |
| append_to_weights(`data`) | Append portfolio daily weight to database during backtesting  | Dataframe: `data` |
| append_to_details(`data`) | Append run details to database during backtesting  | Dataframe: `data` |
| get_last_id() | Get last row id from database  | NA |
| get_table_data() | Get database table in dataframe format  | NA |

# Code example
* Initialization
```
# Import package
from helperClass.genericfunction import *
from helperClass.stockdata import StockData
import warnings
warnings.filterwarnings('ignore')

# Fix seed for comparability and reproducability
random.seed(99)

# Create StockData object
Stock_Data = StockData()

# Set period
Stock_Data.train_start = "2020-01-01"
Stock_Data.train_end = "2022-12-31"
Stock_Data.test_end = "2023-04-01"

# Initialize general function object
gf = GenericFunction(Stock_Data)
```

* Database management
```
# Reset database (Optional) 
database.reset_table("Weights")
database.reset_table("Details")
```

* methods_metrics()
```
# Set necessary properties
gf.set_disable_tqdm(False)           --- Optional
gf.gamma2 = 0.001                    --- Set gamma^2 for MVP
gf.aggregate_method = "geo_median"   --- Set aggregation method
gf.boot_properties = 50              --- Set when you run bootstrapping
gf.set_ssr_properties(50,15)         --- Set when you run subset resampling 
gf.set_ensemble_properties(50,252,50,15)  --- Set when you run ensemble strategies
gf.set_n_day_rebalance(1)            --- Rebalance period

# Perform strategy
gf.methods_metrics("boot_ssr_mvp")
```

* Expected output
```
Run details have been saved.
Number of Resamples = 50, Size of Each Resample = 252, Number of Resampled Subsets = 50, Size of Each Subset = 15 || return: 0.025815166781349985, sharpe: 0.6100607183994026, Volatility: 0.20405974071292862, Turnover Rate: 0.026181505281910118, Maximum Drawdown: -0.10800116687895411
```

* plot_plot_histogram()
```
# Set necessary properties
gf.set_disable_tqdm(False)           --- Optional
gf.gamma2 = 0.001                    --- Set gamma^2 for MVP
gf.aggregate_method = "geo_median"   --- Set aggregation method
gf.boot_properties = 50              --- Set when you run bootstrapping
gf.set_ssr_properties(50,15)         --- Set when you run subset resampling 
gf.set_ensemble_properties(50,252,50,15)  --- Set when you run ensemble strategies
gf.set_n_day_rebalance(1)            --- Rebalance period

# Perform plotting
gf.plot_histogram("ssr",50,False)
```

* plot_random_time()
```
# Set necessary properties
gf.set_disable_tqdm(False)           --- Optional
gf.gamma2 = 0.001                    --- Set gamma^2 for MVP
gf.aggregate_method = "geo_median"   --- Set aggregation method
gf.boot_properties = 50              --- Set when you run bootstrapping
gf.set_ssr_properties(50,15)         --- Set when you run subset resampling 
gf.set_ensemble_properties(50,252,50,15)  --- Set when you run ensemble strategies
gf.set_n_day_rebalance(1)            --- Rebalance period

# Perform plotting
gf.plot_random_time("ssr",30,365,30,False)
```

* plot_random_time_random_stock()
```
# Set necessary properties
gf.set_disable_tqdm(False)           --- Optional
gf.gamma2 = 0.001                    --- Set gamma^2 for MVP
gf.aggregate_method = "geo_median"   --- Set aggregation method
gf.boot_properties = 50              --- Set when you run bootstrapping
gf.set_ssr_properties(50,15)         --- Set when you run subset resampling 
gf.set_ensemble_properties(50,252,50,15)  --- Set when you run ensemble strategies
gf.set_n_day_rebalance(1)            --- Rebalance period

# Perform plotting
gf.plot_random_time_random_stock("boot_ssr_mvp",50,365,365,n,'helperData/stock_time_2024-04-03_00_22_28_915340.pickle',False)
```

* Check the results
```
# Set necessary properties
database.get_table_data("Details")
database.get_table_data("Weights")
```



