win2 = 14
lb2 = 30
mb2 = 50
ub2 = 70

def rsi(x, n=14):
    U = np.maximum(x, 0)
    D = -1 * np.minimum(x, 0)
    RS = U.rolling(n).mean() / D.rolling(n).mean()
    return 100 - 100 / (1 + RS)
    
    
    
    
    
tsla2['RSI'] = rsi(x=tsla2['Adj Close'].diff(), n=win2)

tsla2['Position'] = np.select(
    condlist=[
        (tsla2['RSI'].shift(1) > lb2) & (tsla2['RSI'].shift(2) <= lb2),
        (tsla2['RSI'].shift(1) > mb2) & (tsla2['RSI'].shift(2) <= mb2),
        (tsla2['RSI'].shift(1) < ub2) & (tsla2['RSI'].shift(2) >= ub2),
        (tsla2['RSI'].shift(1) < mb2) & (tsla2['RSI'].shift(2) >= mb2)
    ], 
    choicelist=[1, 0, -1, 0],
    default=np.nan
)
tsla2['Position'].fillna(method='ffill', inplace=True)
tsla2['Position'].fillna(value=0, inplace=True)

tsla2.eval('R_Strategy = Position * R_TSLA', inplace=True)

_ = tsla2.loc['2020', ['R_TSLA', 'R_Strategy']].dropna()
_.add(1).cumprod().plot()
plt.ylabel('Value ($)')
plt.legend(['Buy-and-Hold', 'RSI(14)'])
_ = (_.dropna().index[0] - pd.offsets.BDay(1)).strftime('%B %d, %Y')
plt.title('Value of $1 Invested in TSLA at Close on ' + _)
plt.show()  
