We can use `.groupby()` and `.first()` to keep the first observation per group.
However, we should be careful that our data are "correctly" sorted.

dup_ts.groupby(level=0).first()

dup_ts.groupby(level=0).mean()

What if we wanted the first ***two*** observations per group?

dup_ts.groupby(level=0).head(2)

What if we wanted the ***largest*** observation per group?

dup_ts.groupby(level=0).max()

What if we wanted the ***largest* TWO** observations per group?

dup_ts.groupby(level=0).nlargest(2)




gme = yf.download(tickers='GME', interval='1m', session=session, period='6d')

gme['Adj Close'].head(11)

gme['Adj Close'].resample('5T', closed='right', label='right').last().pct_change()

The following version with `closed = 'left'`, `label = 'left'`, and `.first()` provides the same results, but adds a final return at the end because it grabs one more closing price at the end of the final day.

gme['Adj Close'].resample('5T', closed='left', label='left').first().pct_change()




spy_goog = yf.download(tickers='SPY GOOG', session=session)

spy_goog_ret = spy_goog['Adj Close'].pct_change()

spy_goog_ret

spy_goog_ret.resample('M').std().dropna().mul(100 * np.sqrt(252)).plot()
plt.ylabel('Annualized Volatility of Daily Returns (%)')
plt.title('Volatility')
plt.show()



numerator = returns.rolling(window=252, min_periods=200).cov(spx_rets)

denominator = spx_rets.rolling(window=252, min_periods=200).var()

betas = numerator.div(denominator, axis=0)

betas.describe()

betas.plot()
plt.xlabel('Date')
plt.ylabel('CAPM $\\beta$')
plt.title('Rolling 252-Trading Day CAPM $\\beta$s')
plt.show()




def sr(x):
    return x.mean() / x.std()

gme = yf.download('GME', session=session)
ff = pdr.get_data_famafrench('F-F_Research_Data_Factors_daily', start='1900', session=session)
gme = gme.join(ff[0] / 100, how='inner')

gme['R'] = gme['Adj Close'].pct_change()
gme.eval('R_RF = R - RF', inplace=True)

gme['SR252'] = np.sqrt(252) * gme['R_RF'].rolling(window=252).apply(sr)

gme['SR252'].plot()
plt.ylabel('Sharpe Ratio')
plt.title('Rolling 252-Trading Day Sharpe Ratios for GME')
plt.show()