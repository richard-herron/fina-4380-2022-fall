def annual_return(r):
    years = (r.index[-1] - (r.index[0] - pd.offsets.BDay(1))).days / 365
    return (1 + r).product() ** (1 / years) - 1

def cumulative_returns(r):
    return (1 + r).product() - 1

def annual_volatility(r):
    return np.sqrt(252) * r.std()

def sharpe_ratio(r):
    return np.sqrt(252) * r.mean() / r.std()

def drawdown(r):
    cumprod = (1 + r).cumprod()
    return cumprod / (cumprod).cummax() - 1

def max_drawdown(r):
    return np.max(np.abs(drawdown(r)))

def calmar_ratio(r):
    return annual_return(r) / max_drawdown(r)

def stability(r):
    df = pd.DataFrame({
        'cumlogr': np.log1p(r).cumsum(),
        't': np.arange(r.shape[0])
    })
    
    from statsmodels.formula.api import ols
    r2 = ols('cumlogr ~ 1 + t', data=df).fit().rsquared
    
    return r2 

def omega_ratio(r, T=0):
    mu = r.mean()
    sigma = r.std()

    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 10000)
    dx = x[1] - x[0]

    from scipy.stats import norm
    c = norm.cdf(x, loc=mu, scale=sigma)
    omc = 1 - c
    
    return (omc[x > T] * dx).sum() / (c[x <= T] * dx).sum()    

def sortino_ratio(r, T=0):
    mu = r.mean()
    sigma = r.std()

    x = np.linspace(mu - 4*sigma, T, 10000)
    dx = x[1] - x[0]
    xmT = x - T
    
    from scipy.stats import norm
    p = norm.pdf(x, loc=mu, scale=sigma)
    
    return np.sqrt(252) * (mu - T) / np.sqrt((xmT * xmT * p * dx).sum())

def skew(r):
    return r.skew()

def kurtosis(r):
    return r.kurt()

def tail_ratio(r, cutoff=0.05):
    tails = np.percentile(r, [100*cutoff, 100*(1 - cutoff)])
    return -1 * tails[1] / tails[0]

def daily_value_at_risk(r, cutoff=0.05):
    return np.percentile(r, 100*cutoff)

def plot_cumulative_returns(r):
    (1 + r).cumprod().plot()
    plt.title('Cumulative Returns on $1 Investment')
    plt.ylabel('Cumulative Returns ($)')
    plt.axhline(1, color='k', linestyle='--')
    plt.show()
    return None

def plot_rolling_sharpe_ratio(r, n=126):
    (np.sqrt(252) * r.rolling(n).mean() / r.rolling(n).std()).plot()
    plt.title('Rolling Sharpe Ratio (' + str(n) + ' Trading Day)')
    plt.ylabel('Sharpe Ratio')
    plt.axhline(sharpe_ratio(r), color='k', linestyle='--')
    plt.legend(['Rolling', 'Mean'])
    plt.show()
    return None

def plot_underwater(r):
    (100 * drawdown(r)).plot()
    plt.title('Underwater Plot')
    plt.ylabel('Drawdown (%)')
    plt.show()
    return None





def tear_sheet(r):
    dic = {
        'Annual Return': annual_return(r),
        'Cumulative Returns': cumulative_returns(r),
        'Annual Volatility': annual_volatility(r),
        'Sharpe Ratio': sharpe_ratio(r),
        'Calmar Ratio': calmar_ratio(r),
        'Stability': stability(r),
        'Max Drawdown': max_drawdown(r),
        'Omega Ratio': omega_ratio(r),
        'Sortino Ratio': sortino_ratio(r),
        'Skew': skew(r),
        'Kurtosis': kurtosis(r),
        'Tail Ratio': tail_ratio(r),
        'Daily Value at Risk': daily_value_at_risk(r)
    }
    df = pd.DataFrame(data=dic.values(), columns = ['Backtest'], index=dic.keys())
    display(df)
    return None




_ = prices_df['Adj Close'].pct_change().dropna().dot(portfolio_weights)
(1 + _).cumprod().rolling(504).apply(lambda x : x[-1] / x.max() - 1).plot()




portf_results_df.iloc[portf_results_df['volatility'].argmin()]

weights.iloc[portf_results_df['volatility'].argmin()]

portf_results_df.iloc[portf_results_df['sharpe_ratio'].argmax()]

weights.iloc[portf_results_df['sharpe_ratio'].argmax()]




def get_portf_sr_neg(w, avg_rtns, cov_mat):
    return -1 * get_portf_sr(w, avg_rtns, cov_mat)

res = sco.minimize(
    fun=get_portf_sr_neg, 
    args=(avg_returns, cov_mat), 
    x0=weights.mean(),
    constraints=(
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ),
    bounds=tuple((0,1) for i in avg_returns)
)

print(
    'Maximum Sharpe Ratio Portfolio',
    '==============================',
    '',
    'Performance',
    '-----------',
    'Return:       {:0.4f}'.format(get_portf_rtn(res['x'], avg_returns)),
    'Volatility:   {:0.4f}'.format(get_portf_vol(res['x'], cov_mat)),
    'Sharpe Ratio: {:0.4f}'.format(get_portf_sr(res['x'], avg_returns, cov_mat)),
    '',
    sep='\n'
)

print('Weights', '-------', sep='\n')
for i, j in zip(returns_df.columns, res['x']):
    print((i + ':').ljust(14) + '{:0.4f}'.format(j))

    
    
    
    
returns_df_ = prices_df.loc['2019':, 'Adj Close'].pct_change().dropna()

avg_returns_ = 252*returns_df_.mean()
cov_mat_ = 252*returns_df_.cov()

print(
    'Maximum Sharpe Ratio Portfolio Out-of-Sample',
    '============================================',
    '',
    'Performance',
    '-----------',
    'Return:       {:0.4f}'.format(get_portf_rtn(res['x'], avg_returns_)),
    'Volatility:   {:0.4f}'.format(get_portf_vol(res['x'], cov_mat_)),
    'Sharpe Ratio: {:0.4f}'.format(get_portf_sr(res['x'], avg_returns_, cov_mat_)),
    '',
    sep='\n'
)

print('Weights', '-------', sep='\n')
for i, j in zip(returns_df_.columns, res['x']):
    print((i + ':').ljust(14) + '{:0.4f}'.format(j))


_ = yf.download('SPY', session=session)['Adj Close'].pct_change().dropna()
np.sqrt(252) * _.loc['2019':].mean() / _.loc['2019':].std()




temp = np.array([-.15, -0.15, 1.3, 0])

res = sco.minimize(
    fun=get_portf_sr_neg, 
    args=(avg_returns, cov_mat), 
    x0=weights.mean(),
    constraints=(
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ),
    bounds=tuple((-0.3, 1.3) for i in avg_returns)
)

print(
    'Maximum Sharpe Ratio Portfolio WITH UP TO 30% SHORT POSITION',
    '============================================================',
    '',
    'Performance',
    '-----------',
    'Return:       {:0.4f}'.format(get_portf_rtn(res['x'], avg_returns)),
    'Volatility:   {:0.4f}'.format(get_portf_vol(res['x'], cov_mat)),
    'Sharpe Ratio: {:0.4f}'.format(get_portf_sr(res['x'], avg_returns, cov_mat)),
    '',
    sep='\n'
)

print('Weights', '-------', sep='\n')
for i, j in zip(returns_df.columns, res['x']):
    print((i + ':').ljust(14) + '{:0.4f}'.format(j))


res = sco.minimize(
    fun=get_portf_sr_neg, 
    args=(avg_returns, cov_mat), 
    x0=weights.mean(),
    constraints=(
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: np.sum(x[x < 0]) + 0.3}
    ),
    bounds=tuple((-0.3, 1.3) for i in avg_returns)
)

print(
    'Maximum Sharpe Ratio Portfolio WITH UP TO 30% SHORT POSITION',
    '============================================================',
    '',
    'Performance',
    '-----------',
    'Return:       {:0.4f}'.format(get_portf_rtn(res['x'], avg_returns)),
    'Volatility:   {:0.4f}'.format(get_portf_vol(res['x'], cov_mat)),
    'Sharpe Ratio: {:0.4f}'.format(get_portf_sr(res['x'], avg_returns, cov_mat)),
    '',
    sep='\n'
)

print('Weights', '-------', sep='\n')
for i, j in zip(returns_df.columns, res['x']):
    print((i + ':').ljust(14) + '{:0.4f}'.format(j))

    
    
    
    
    
df = pd.DataFrame(np.nan, index=range(100), columns=['volatility', 'return'])

i = 0

for tgt in np.linspace(avg_returns.min(), avg_returns.max(), 100):
    res = sco.minimize(
        fun=get_portf_vol, 
        args=(cov_mat), 
        x0=weights.mean(),
        constraints=(
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, # we want this lambda function to evaluate to 0
            {'type': 'eq', 'fun': lambda x: get_portf_rtn(x, avg_returns) - tgt} # we want this lambda function to evaluate to the target return
        ),
        bounds=tuple((0,1) for i in avg_returns)
    )
    
    df['volatility'].iloc[i] = res['fun']
    df['return'].iloc[i] = get_portf_rtn(res['x'], avg_returns)
    
    i += 1

MARKS = {
    'FB': 'o', 
    'MSFT': 'X', 
    'TSLA': 'd', 
    'TWTR': '*'
}

fig, ax = plt.subplots()

df.plot(
    kind='line',
    style='b--',
    label='Efficient Frontier',
    x='volatility',
    y='return',
    ax=ax
)

for s in returns_df.columns:
    ax.scatter(
        x=np.sqrt(cov_mat.loc[s, s]), 
        y=avg_returns.loc[s], 
        marker=MARKS[s], 
        label=s
    )
    
plt.legend()


ax.set(
    xlabel='Volatility', 
    ylabel='Expected Returns', 
    title='Efficient Frontier from sco.minimize()'
)

plt.show()