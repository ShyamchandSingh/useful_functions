INPUT DATA COLUMNS: 

[pnl, exit_date, exit_time, gross_pnl, charges]

pnl: actual profit/loss after charges are subtracted
exit_date: trade closing date
exit_time: trade closing time
gross_pnl: profit/loss before deductions
charges: total applicable charges (brokerage+tax+others) # for 4 lots of NIFTY OPTION instrument 300 can be assumed



RUN FROM PYTHON:

## CSV INPUT
##-------------------------------------------------------------------
from we_v1 import compute_we_from_csv

result = compute_we_from_csv("trades.csv", start_capital=1_000_000)
print(result["Weighted efficiency score (/100)"])

## DATAFRAME INPUT
##-------------------------------------------------------------------
from we_v1 import compute_we_from_dataframe
result = compute_we_from_dataframe(df, start_capital=1_000_000)


## RUN FROM BASH:
##-------------------------------------------------------------------
python we_v1.py trades.csv
