# ==========================================================
# 5.e — TRADE RELATIONSHIP EXPLORER
# ==========================================================

st.subheader("5.e Trade Relationship Explorer (Correlation + Lag + Granger)")

from statsmodels.tsa.stattools import grangercausalitytests

try:

    # build derivative time series
    def extract_original_columns(df):
        if df is None or df.empty:
            return pd.DataFrame()
        cols = [c for c in df.columns if "(Original)" in c]
        clean = df[cols].copy()
        clean.columns = [c.replace(" (Original)", "") for c in cols]
        return clean

    derivatives_ts = pd.concat([
        extract_original_columns(historical_spreads_3M_df),
        extract_original_columns(historical_butterflies_3M_df),
        extract_original_columns(historical_double_butterflies_3M_df),
        extract_original_columns(historical_spreads_6M_df),
        extract_original_columns(historical_butterflies_6M_df),
        extract_original_columns(historical_double_butterflies_6M_df),
        extract_original_columns(historical_spreads_12M_df),
        extract_original_columns(historical_butterflies_12M_df),
        extract_original_columns(historical_double_butterflies_12M_df),
    ], axis=1).dropna(axis=1, how="all")

    if derivatives_ts.empty:
        st.stop()

    # rolling lookback
    total_days = len(derivatives_ts)
    lookback_days = st.slider("Rolling Lookback Days", 30, total_days, min(250,total_days))
    derivatives_ts = derivatives_ts.tail(lookback_days)

    trade_selected = st.selectbox("Select Trade", sorted(derivatives_ts.columns))

    col1, col2, col3 = st.columns(3)
    with col1:
        corr_threshold = st.slider("Min |Correlation|",0.0,1.0,0.5,0.05)
    with col2:
        max_lag_days = st.slider("Max Lag Days",1,20,5)
    with col3:
        granger_threshold = st.slider("Max Granger p-value",0.01,1.0,0.05,0.01)

    # FFT lag
    def compute_lead_lag_fft(x,y,max_lag):
        df = pd.concat([x,y],axis=1).dropna()
        if len(df)<50: return None,0
        x=(df.iloc[:,0]-df.iloc[:,0].mean())/df.iloc[:,0].std()
        y=(df.iloc[:,1]-df.iloc[:,1].mean())/df.iloc[:,1].std()
        corr=np.correlate(x,y,mode="full")
        lags=np.arange(-len(x)+1,len(x))
        mask=(lags>=-max_lag)&(lags<=max_lag)
        corr,lags=corr[mask],lags[mask]
        i=np.argmax(np.abs(corr))
        return corr[i]/len(x),lags[i]

    # granger
    def granger_test(x,y,max_lag):
        df=pd.concat([x,y],axis=1).dropna()
        if len(df)<100: return None
        try:
            res=grangercausalitytests(df,maxlag=max_lag,verbose=False)
            return min([res[i+1][0]["ssr_ftest"][1] for i in range(max_lag)])
        except:
            return None

    trade_series=derivatives_ts[trade_selected].dropna()
    results=[]

    for col in derivatives_ts.columns:
        if col==trade_selected: continue

        other=derivatives_ts[col]
        idx=trade_series.index.intersection(other.index)
        if len(idx)<100: continue

        t=trade_series.loc[idx]
        o=other.loc[idx]

        corr=t.corr(o)
        lag_corr,lag=compute_lead_lag_fft(t,o,max_lag_days)
        pval=granger_test(t,o,max_lag_days)

        relation="Trade Leads" if lag>0 else "Trade Follows" if lag<0 else "Simultaneous"

        results.append({
            "Trade":trade_selected,
            "Instrument":col,
            "Correlation":corr,
            "Lag":lag,
            "Relationship":relation,
            "Granger p":pval,
            "AbsCorr":abs(corr)
        })

    df=pd.DataFrame(results)

    filtered=df[
        (df["AbsCorr"]>=corr_threshold)&
        ((df["Granger p"].isna())|(df["Granger p"]<=granger_threshold))
    ].sort_values("AbsCorr",ascending=False)

    st.dataframe(filtered.drop(columns="AbsCorr"),use_container_width=True)

except Exception as e:
    st.warning(f"Relationship analysis unavailable: {e}")
