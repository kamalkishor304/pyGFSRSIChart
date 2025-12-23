import streamlit as st
import pandas as pd
from nsetools import Nse
import plotly.express as px
import time
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nsepython import nsefetch
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Equity Portfolio",
    page_icon="📈",
    layout="wide"
)

# ---------------- SHEETS ----------------
SHEETS = [
    "Aishwarya",
    "Sudhadevi",
    "Ramkishor",
    "Rashmi",
    "Dashrathlal"
]

def load_all_trades():
    dfs = []
    for sheet in SHEETS:
        df = pd.read_excel("Equity Trades.xlsx", sheet_name=sheet)
        df['Buy Date'] = df['Buy Date'].fillna(pd.Timestamp('2025-01-01'))
        df = df[['SYMBOL', 'Holder', 'Buy Date', 'Sell', 'Buy', 'QTY']]
        df = df.dropna(thresh=df.shape[1] - 2)
        df['SourceSheet'] = sheet
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    rename_dict = {
        'DHAN': 'Sudhadevi',
        'RAM': 'RamKishor',
        'ANGELONE': 'Dashrathlal'
    }

    all_df['Holder'] = all_df['Holder'].replace(rename_dict)
    all_df['Holder'] = all_df['Holder'].astype(str).str.strip().str.title()
    return all_df

nse = Nse()
holdings = load_all_trades()['SYMBOL'].unique()
holdings_info = {}

def fetch_quote(symbol):
    try:
        temp = nse.get_quote(symbol)
        return symbol, temp['lastPrice'], temp['change']
    except Exception:
        return symbol, None, None


# ⚠️ Keep workers low to avoid NSE blocking
MAX_WORKERS = 16

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(fetch_quote, symbol) for symbol in holdings]

    for future in tqdm(as_completed(futures), total=len(futures)):
        symbol, price, change = future.result()
        if price is not None:
            holdings_info[symbol] = [price, change]
        else:
            print(f"Failed: {symbol}")

print(holdings_info)





# nse = Nse()

def color_unrealised_pl(val):
    try:
        val = float(str(val).replace('₹', '').replace(',', ''))
        if val < 0:
            return "color: red; font-weight: bold"
        elif val > 0:
            return "color: green; font-weight: bold"
    except:
        pass
    return ""


def highlight_pl_bg(val):
    try:
        val = float(str(val).replace('₹', '').replace(',', ''))
        if val < 0:
            return "background-color: #ffcccc"   # light red
        elif val > 0:
            return "background-color: #ccffcc"   # light green
    except:
        pass
    return ""



# ---------------- TREND ANALYSIS CORE ----------------
def get_monthly_rsi_trend(symbol): 
    try: 
        df = yf.download( 
            f"{symbol}.NS", 
            period="10y", 
            interval="1mo", # ✅ TRUE monthly data 
            auto_adjust=True, 
            progress=False 
        ) 
        if df.empty or df.shape[0] < 20: 
            return None 
        # breakpoint() 
        monthly_close = df['Close'].dropna() 
        monthly_rsi = calculate_rsi(monthly_close) 
        monthly_rsi_ema = calculate_ema(monthly_rsi) 
        date = monthly_rsi_ema.index 
        monthly_rsi = [float(i[0]) for i in monthly_rsi.values] 
        monthly_rsi_ema = [float(i[0]) for i in monthly_rsi_ema.values] 
        
        # ✅ DROP NaNs BEFORE COMPARISON 
        combined = pd.DataFrame({ "Date":date, "RSI": monthly_rsi, "EMA": monthly_rsi_ema }) 
        if combined.empty: 
            return None 
        latest = combined.iloc[-1] 
        prev = combined.iloc[-2]
        return { 
             "SYMBOL": symbol, 
             "Monthly RSI": round(latest["RSI"], 2), 
             "Monthly RSI EMA": round(latest["EMA"], 2), 
             "Prev RSI": round(prev["RSI"], 2) if prev is not None else None, 
             "Prev EMA": round(prev["EMA"], 2) if prev is not None else None, 
             } 
    except Exception as e: 
        return None


# ================= RSI FUNCTIONS =================
def calculate_ema(data, window=21):
    return data.ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_index_stocks(stocks):
    stock_list = []
    if stocks and 'data' in stocks:
        for stock in stocks['data']:
            stock_list.append(stock['symbol'])
    return sorted(stock_list)


def get_index_symbols(index_name):
    data = nsefetch(
        f"https://www.nseindia.com/api/equity-stockIndices?index={index_name}"
    )
    return sorted([item["symbol"] for item in data["data"]])

@st.cache_data(ttl=3600)
def fetch_stock_list():
    try:
        nifty_largemidcap250 = nsefetch(
            'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20LARGEMIDCAP%20250'
        )
        return get_index_stocks(nifty_largemidcap250)
    except:
        return ["RELIANCE", "TCS", "INFY"]

# ---------------- FUNCTIONS ----------------
@st.cache_data(show_spinner=False)
def get_ltp(symbol):
    return holdings_info[symbol][0] if symbol in holdings_info else None

def get_change(symbol):
    return holdings_info[symbol][1] if symbol in holdings_info else None


@st.cache_data(show_spinner=False)
def load_all_trades():
    dfs = []
    for sheet in SHEETS:
        df = pd.read_excel("Equity Trades.xlsx", sheet_name=sheet)
        df['Buy Date'] = df['Buy Date'].fillna(pd.Timestamp('2025-01-01'))
        df = df[['SYMBOL', 'Holder', 'Buy Date', 'Sell', 'Buy', 'QTY']]
        df = df.dropna(thresh=df.shape[1] - 2)
        df['SourceSheet'] = sheet
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    rename_dict = {
        'DHAN': 'Sudhadevi',
        'RAM': 'RamKishor',
        'ANGELONE': 'Dashrathlal'
    }

    all_df['Holder'] = all_df['Holder'].replace(rename_dict)
    all_df['Holder'] = all_df['Holder'].astype(str).str.strip().str.title()
    return all_df


def enrich_portfolio(df):
    df = df.copy()

    # ❗ Exclude sold stocks
    df = df[(df['Sell'].isna()) | (df['Sell'] == 0)]

    total = len(df)
    ltp_list = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, symbol in enumerate(df['SYMBOL']):
        ltp = get_ltp(symbol)
        ltp_list.append(ltp)
        progress_bar.progress((idx + 1) / total)
        status_text.text(f"Fetching LTP for {symbol} ({idx+1}/{total})")
        time.sleep(0.005)

    df['LTP'] = ltp_list
    df['Invested'] = df['Buy'] * df['QTY']
    df['Current'] = df['LTP'] * df['QTY']
    df['P&L'] = df['Current'] - df['Invested']
    df['P&L %'] = (df['P&L'] / df['Invested']) * 100

    progress_bar.empty()
    status_text.empty()
    return df


def portfolio_summary(df):
    invested = df['Invested'].sum()
    current = df['Current'].sum()
    profit = current - invested
    return invested, current, profit


def portfolio_pie_by_stocks(df):
    pie_df = df.groupby('SYMBOL', as_index=False)['Current'].sum()

    fig = px.pie(
        pie_df,
        names='SYMBOL',
        values='Current',
        hole=0.6,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_traces(textinfo="percent+label", textposition='inside')
    fig.update_layout(height=400)
    return fig


def portfolio_treemap(df):
    fig = px.treemap(
        df,
        path=['SYMBOL'],
        values='Current',
        color='P&L',
        color_continuous_scale='RdYlGn',
        custom_data=[
            'Invested',
            'Current',
            'P&L',
            'P&L %',
            'QTY'
        ]
    )

    fig.update_traces(
        texttemplate=(
            "<b>%{label}</b><br>"
            "₹%{customdata[1]:,.0f}<br>"
            "P&L: ₹%{customdata[2]:,.0f}<br>"
            "%{customdata[3]:.2f}%"
        ),
        textinfo="label+text"
    )

    fig.update_layout(
        height=500,
        margin=dict(t=30, l=10, r=10, b=10)
    )

    return fig


# 🔥 ONLY ADDITION — STYLING FUNCTION
def style_pl_df(df):
    def color_pl(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return "color: green; font-weight: bold"
            elif val < 0:
                return "color: red; font-weight: bold"
        return ""

    return df.style.applymap(color_pl, subset=["P&L", "P&L %"])


# ---------------- LOAD DATA ----------------
df_all = load_all_trades()
symbols = df_all['SYMBOL'].dropna().unique().tolist()


holders = sorted(df_all['Holder'].unique())

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to",
    # ["🏠 Home", "👤 Individual Portfolio", "📑 Pivot Data", "📉 RSI Analysis"]
    ["🏠 Home", "👤 Individual Portfolio", "📑 Pivot Data", "📉 RSI Analysis", "📊 Trend Analysis"]

)




if "selected_holder" not in st.session_state:
    st.session_state.selected_holder = holders[0]

# ================= HOME =================
if page == "🏠 Home":
    st.title("🏠 Portfolio Overview")
    st.caption("All holders at a glance")

    chunk_size = 2
    for i in range(0, len(holders), chunk_size):
        chunk = holders[i:i + chunk_size]
        cols = st.columns(len(chunk))

        for col, holder in zip(cols, chunk):
            with col:
                df_holder = df_all[df_all['Holder'] == holder]
                with st.spinner(f"Enriching portfolio for {holder}..."):
                    df_holder = enrich_portfolio(df_holder)

                invested, current, profit = portfolio_summary(df_holder)

                st.subheader(holder)
                st.metric(
                    "P&L",
                    f"₹ {profit:,.0f}",
                    delta=f"{(profit / invested) * 100:.2f}%" if invested else "0%"
                )

                st.plotly_chart(
                    portfolio_pie_by_stocks(df_holder),
                    use_container_width=True
                )

                if st.button(f"View {holder}", key=f"btn_{holder}"):
                    st.session_state.selected_holder = holder
                    st.experimental_rerun()

# ================= INDIVIDUAL =================
elif page == "👤 Individual Portfolio":
    st.title("👤 Individual Portfolio")

    selected_holder = st.selectbox(
        "Select Holder",
        holders,
        index=holders.index(st.session_state.selected_holder)
    )

    st.session_state.selected_holder = selected_holder
    df_person = df_all[df_all['Holder'] == selected_holder]

    with st.spinner(f"Enriching portfolio for {selected_holder}..."):
        df_person = enrich_portfolio(df_person)

    invested, current, profit = portfolio_summary(df_person)
    profit_pct = (profit / invested) * 100 if invested else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Invested", f"₹ {invested:,.0f}")
    c2.metric("📈 Current", f"₹ {current:,.0f}")
    c3.metric("P&L", f"₹ {profit:,.0f}", delta=f"{profit_pct:.2f}%")
    c4.metric("📊 Stocks", df_person.shape[0])

    chart_type = st.radio(
        "Chart Type",
        ["Pie Chart", "Treemap"],
        horizontal=True
    )

    if chart_type == "Pie Chart":
        st.plotly_chart(
            portfolio_pie_by_stocks(df_person),
            use_container_width=True
        )
    else:
        st.plotly_chart(
            portfolio_treemap(df_person),
            use_container_width=True
        )

    # 🔥 ONLY CHANGE HERE (Styled DF)

    styled_df = (
    df_person.style
        .format({
            'Buy Date': '{:%Y-%m-%d}',
            'Buy': '₹{:,.2f}',
            'Sell': '₹{:,.2f}',
            'QTY': '{:,.0f}',
            'LTP': '₹{:,.2f}',    
            'Invested': '₹{:,.2f}',
            'Current': '₹{:,.2f}',
            'P&L': '₹{:,.2f}',
            'P&L %': '{:.2f}%',            
        })
        .applymap(highlight_pl_bg, subset=['P&L'])
        .applymap(highlight_pl_bg, subset=['P&L %'])
)
    st.dataframe(styled_df, use_container_width=True)
    # st.dataframe(style_pl_df(df_person), use_container_width=True)

# ================= PIVOT DATA =================
elif page == "📑 Pivot Data":
    st.title("📑 Pivot Data")

    selected_holder = st.selectbox("Select Holder", ["All Holders"] + holders)
    df_pivot = df_all.copy() if selected_holder == "All Holders" else df_all[df_all['Holder'] == selected_holder]

    df_pivot['Change'] = df_pivot['SYMBOL'].apply(get_change)
    df_pivot['LTP'] = df_pivot['SYMBOL'].apply(get_ltp)

    df_pivot['Realised P/L'] = df_pivot.apply(
        lambda r: (r['Sell'] - r['Buy']) * r['QTY']
        if pd.notna(r['Sell']) and r['Sell'] > 0 else 0,
        axis=1
    )

    df_pivot['Unrealised P/L'] = df_pivot.apply(
        lambda r: (r['LTP'] - r['Buy']) * r['QTY']
        if pd.isna(r['Sell']) or r['Sell'] == 0 else 0,
        axis=1
    )

    grouped = df_pivot.groupby('SYMBOL', as_index=False).agg({
        'Realised P/L': 'sum',
        'Unrealised P/L': 'sum',
        'Buy': lambda x: (x * df_pivot.loc[x.index, 'QTY']).sum() / df_pivot.loc[x.index, 'QTY'].sum(),
        'QTY': 'sum',
        'Change': 'last'
    })

    grouped['Change Today'] = grouped.apply(
    lambda r: 0 if r['Unrealised P/L'] == 0 else r['Change'] * r['QTY'],
    axis=1
)



    df_display = grouped[['SYMBOL', 'Realised P/L', 'Unrealised P/L', 'Change Today']]
    # df_display['Realised P/L'] = df_display['Realised P/L'].map('₹{:,.2f}'.format)
    # df_display['Unrealised P/L'] = df_display['Unrealised P/L'].map('₹{:,.2f}'.format)
    # df_display['Change Today'] = df_display['Change Today'].map('₹{:,.2f}'.format)

    # st.dataframe(df_display, use_container_width=True)
        # ================= BAR CHART: TODAY'S CHANGE BY HOLDER =================
        # ================= BAR CHART: TODAY'S CHANGE (UNREALISED ONLY) =================
    st.subheader("📊 Today's Change by Holder (Unrealised Positions Only)")

    # Filter only unrealised positions
    bar_df = df_pivot[
        (df_pivot['Unrealised P/L'] != 0)
    ].copy()

    if not bar_df.empty:
        # Calculate today's change in ₹ (qty weighted)
        bar_df['Today Change ₹'] = (bar_df['Change'] * bar_df['QTY']) 

        holder_change = (
            bar_df
            .groupby('Holder', as_index=False)['Today Change ₹']
            .sum()
        )
        holder_change['Direction'] = holder_change['Today Change ₹'].apply(
    lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Zero'
)
        holder_change = holder_change.sort_values(
        by='Holder',
        ascending=True
        )



        fig_bar = px.bar(
            holder_change,
            x='Holder',
            y='Today Change ₹',
            color='Direction',
            text='Today Change ₹',
            title="Sum of Today's Change (₹) by Holder (Unrealised Only)",
            labels={
                'Holder': 'Holder',
                'Today Change ₹': '₹ Change'
            },
            color_discrete_map={
                'Positive': 'green',
                'Negative': 'red',
                'Zero': 'gray'
            }
        )
        fig_bar.update_traces(
    texttemplate='₹ %{text:,.2f}',
    textposition='inside'
)



        fig_bar.update_layout(
            xaxis=dict(
                categoryorder='array',
                categoryarray=holder_change['Holder']
            ),
            xaxis_tickangle=-30,
            height=450,
            template='plotly_white'
        )


        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No unrealised positions available for today's change.")


    styled_df = (
    df_display.style
        .format({
            'Realised P/L': '₹{:,.2f}',
            'Unrealised P/L': '₹{:,.2f}',
            'Change Today': '₹{:,.2f}'
        })
        .applymap(highlight_pl_bg, subset=['Change Today'])
        .applymap(color_unrealised_pl, subset=['Unrealised P/L'])
)

    st.dataframe(styled_df, use_container_width=True)




# ================= RSI ANALYSIS =================
elif page == "📉 RSI Analysis":
    st.title("📉 Interactive Stock RSI Plotter")
    st.caption("Daily, Weekly & Monthly RSI with EMA overlays")

    stock_list = fetch_stock_list()
    selected_stock = st.selectbox("Select Stock", ["Select"] + stock_list)

    if selected_stock != "Select":
        with st.spinner("Fetching data..."):
            try:
                ticker = selected_stock.strip().upper()
                df = yf.download(
                    f"{ticker}.NS",
                    period='15y',
                    interval='1d',
                    auto_adjust=True
                )

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                data = df.dropna().copy()
                data['RSI'] = calculate_rsi(data['Close'])
                data['EMA_RSI'] = calculate_ema(data['RSI'])

                weekly_close = data['Close'].resample('W').last()
                weekly_ema50 = calculate_ema(weekly_close, 50)
                weekly_rsi = calculate_rsi(weekly_close)
                weekly_rsi_ema = calculate_ema(weekly_rsi, 50)

                monthly_close = data['Close'].resample('M').last()
                monthly_rsi = calculate_rsi(monthly_close)
                monthly_rsi_ema = calculate_ema(monthly_rsi)

                

                fig = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    row_heights=[0.35, 0.2, 0.2, 0.2],
                    subplot_titles=(
                        f'{ticker} Weekly Close Price',
                        'Monthly RSI',
                        'Weekly RSI',
                        'Daily RSI'
                    )
                )

                fig.add_trace(
                    go.Scatter(x=weekly_close.index, y=weekly_close,
                               name='Weekly Close', line=dict(width=3)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=weekly_ema50.index, y=weekly_ema50,
                               name='50 EMA', line=dict(width=2)),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=monthly_rsi.index, y=monthly_rsi,
                               name='Monthly RSI'),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=monthly_rsi_ema.index, y=monthly_rsi_ema,
                               name='EMA RSI'),
                    row=2, col=1
                )

                fig.add_trace(
                    go.Scatter(x=weekly_rsi.index, y=weekly_rsi,
                               name='Weekly RSI'),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=weekly_rsi_ema.index, y=weekly_rsi_ema,
                               name='EMA RSI'),
                    row=3, col=1
                )

                fig.add_trace(
                    go.Scatter(x=data.index, y=data['EMA_RSI'],
                               name='Daily RSI EMA'),
                    row=4, col=1
                )

                for r in [2, 3, 4]:
                    fig.add_hline(y=40, line_dash='dot', row=r, col=1,line_color="red")
                    fig.add_hline(y=50, line_dash='dot', row=r, col=1)
                    fig.add_hline(y=60, line_dash='dot', row=r, col=1,line_color="blue")
                    fig.update_yaxes(range=[0, 100], row=r, col=1)

                fig.update_layout(
                    height=1400,
                    hovermode='x unified',
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error fetching data for {ticker}: {e}")





# ================= TREND ANALYSIS =================
elif page == "📊 Trend Analysis":
    st.title("📊 Monthly RSI Trend Analysis")

    analysis_mode = st.selectbox(
        "Select Trend Analytics Mode",
        [
            "Trend Strength (Strong / Weak)",
            "RSI Slope (Momentum)",
            "Trend Flip Detection"
        ]
    )
    st.subheader("📌 Select Stock Universe")

    stock_universe = st.selectbox(
        "Analyze stocks from",
        [
            "Current Holding",
            "Nifty 50",
            "Nifty Next 50",
            "Nifty Midcap 150",
            "Nifty LARGEMIDCAP 250"
        ]
    )


    if stock_universe == "Current Holding":
        symbols = sorted(
            df_all.loc[df_all['Sell'] == 0, 'SYMBOL']
            .dropna()
            .unique()
        )

    elif stock_universe == "Nifty 50":
        symbols = get_index_symbols("NIFTY 50")

    elif stock_universe == "Nifty Next 50":
        symbols = get_index_symbols("NIFTY%20NEXT%2050")

    elif stock_universe == "Nifty Midcap 150":
        symbols = get_index_symbols("NIFTY MIDCAP 150")

    elif stock_universe == "Nifty LARGEMIDCAP 250":
        symbols = get_index_symbols("NIFTY%20LARGEMIDCAP%20250")

    results = []
    trend_up, trend_down = [], []
    slope_improving, slope_weakening = [], []

    progress = st.progress(0)
    status = st.empty()

    # ---------- WORKER FUNCTION ----------
    def analyze_symbol(symbol):
        data = get_monthly_rsi_trend(symbol)
        if not data:
            return None

        rsi = data["Monthly RSI"]
        ema = data["Monthly RSI EMA"]
        prev_rsi = data["Prev RSI"]
        prev_ema = data["Prev EMA"]

        output = []

        # -------- Trend Strength --------
        if analysis_mode == "Trend Strength (Strong / Weak)":
            if rsi >= 60 and rsi > ema:
                label = "🟢 Strong Uptrend"
            elif rsi >= 50 and rsi < ema:
                label = "🟡 Weak Uptrend"
            elif rsi >= 40 and rsi < ema:
                label = "🟠 Weak Downtrend"
            elif rsi < 40 and rsi < ema:
                label = "🔴 Strong Downtrend"
            else:
                label = "⚪ Neutral"

            output.append({
                "SYMBOL": symbol,
                "RSI": rsi,
                "Trend": label
            })

        # -------- RSI Slope --------
        elif analysis_mode == "RSI Slope (Momentum)":
            slope = rsi - prev_rsi
            label = "🟢 Improving" if slope > 0 else "🔴 Weakening"
            output.append({
                "SYMBOL": symbol,
                "RSI Slope": round(slope, 2),
                "Momentum": label
            })

        # -------- Trend Flip --------
        elif analysis_mode == "Trend Flip Detection":
            prev_trend = "Uptrend" if prev_rsi > prev_ema else "Downtrend"
            curr_trend = "Uptrend" if rsi > ema else "Downtrend"

            if prev_trend != curr_trend:
                output.append({
                    "SYMBOL": symbol,
                    "Previous": prev_trend,
                    "Current": curr_trend,
                    "Flip": "🚀 Bullish" if curr_trend == "Uptrend" else "⚠️ Bearish"
                })

        # -------- RSI Direction --------
        if rsi >= 50:
            trend_up.append({"SYMBOL": symbol, "Monthly RSI": rsi})
        else:
            trend_down.append({"SYMBOL": symbol, "Monthly RSI": rsi})

        # -------- RSI Slope Lists --------
        slope = rsi - prev_rsi
        if slope > 0:
            slope_improving.append({"SYMBOL": symbol, "RSI Slope": round(slope, 2)})
        elif slope < 0:
            slope_weakening.append({"SYMBOL": symbol, "RSI Slope": round(slope, 2)})

        return output
    
    for i, symbol in enumerate(symbols):
        status.text(f"Analyzing {symbol} ({i+1}/{len(symbols)})")
        # print(i)
        result = analyze_symbol(symbol)
        # print(result)

        if result:
            results.extend(result)

        progress.progress((i + 1) / len(symbols))
    

    # ---------- PARALLEL EXECUTION ----------
    #with ThreadPoolExecutor(max_workers=1) as executor:
    #    futures = {executor.submit(analyze_symbol, sym): sym for sym in symbols}
    #
    #    for i, future in enumerate(as_completed(futures)):
    #        symbol = futures[future]
    #       status.text(f"Analyzing {symbol} ({i+1}/{len(symbols)})")

     #       res = future.result()
     #       if res:
     #           results.extend(res)
#
      #      progress.progress((i + 1) / len(symbols))

    progress.empty()
    status.empty()

    # ---------- DISPLAY ----------
    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("No results for selected analytics mode.")