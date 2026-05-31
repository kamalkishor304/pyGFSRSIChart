import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import yfinance as yf
from typing import List

# modular imports
from data import fetch_monthly_close, fetch_quotes, get_ltp_and_change
from analysis import calculate_rsi, calculate_ema, get_monthly_rsi_trend
from portfolio import load_all_trades, get_holdings_quotes, enrich_portfolio_df, SHEETS
# ---------------- CONFIG ----------------
st.set_page_config(page_title="Equity Portfolio", page_icon="📈", layout="wide")

# ---------------- SHEETS ----------------
# load trades and fetch holdings via yfinance
df_all = load_all_trades()
holdings = df_all['SYMBOL'].dropna().unique().tolist()
holdings_info = get_holdings_quotes(holdings, max_workers=12)

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


# Minimal fallback lists to guarantee function returns
DEFAULT_STOCK_LIST = ["RELIANCE", "TCS", "INFY", "HDFC", "ITC"]

NSE_INDICES_FALLBACK = {
    "NIFTY 50": DEFAULT_STOCK_LIST,
    "NIFTY NEXT 50": ["ACC", "AIAENG", "AMBUJACEM"],
    "NIFTY MIDCAP 150": ["3MINDIA", "ABCAPITAL"],
    "NIFTY LARGEMIDCAP 250": DEFAULT_STOCK_LIST
}


@st.cache_data(ttl=3600)
def fetch_stock_list() -> List[str]:
    return DEFAULT_STOCK_LIST



# ================= RSI FUNCTIONS =================
@st.cache_data(ttl=3600)
def get_nse_index_symbols(index_name: str):
    """Dynamically fetch index constituents from NSE with intelligent fallback."""
    import requests    
    # normalize index name for lookups
    normalized_name = index_name.replace("%20", " ")
    try:
        nse_url = f"https://www.nseindia.com/api/equity-stock-indices?index={index_name}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(nse_url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                symbols = [item.get('symbol', '').strip() for item in data['data'] if item.get('symbol')]
                if symbols:
                    return sorted(symbols)
    except Exception as e:
        print(f"Error fetching index symbols for {index_name} from NSE API: {e}")
        st.error(f"Error fetching index symbols for {index_name}: {e}")
    
    # Ensure we always return a list (avoid returning None which breaks static analysis)
    try:
        # final fallback to hardcoded list if present
        return NSE_INDICES_FALLBACK.get(normalized_name, fetch_stock_list())
    except Exception:
        return fetch_stock_list()
    
def get_index_symbols(index_name: str):
    """Get symbols for the given index."""
    return get_nse_index_symbols(index_name)

# ---------------- FUNCTIONS ----------------
@st.cache_data(show_spinner=False)
def get_ltp(symbol):
    return holdings_info.get(symbol, (None, None))[0]

def get_change(symbol):
    return holdings_info[symbol][1] if symbol in holdings_info else None


@st.cache_data(show_spinner=False)
def load_all_trades_cached(path="Equity Trades.xlsx"):
    return load_all_trades(path)


def enrich_portfolio(df):
    # UI-friendly wrapper that uses portfolio.enrich_portfolio_df
    df = df.copy()
    holdings = df['SYMBOL'].dropna().unique().tolist()
    quotes = get_holdings_quotes(holdings, max_workers=12)
    return enrich_portfolio_df(df, quotes)


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
                    width='stretch'
                )

                if st.button(f"View {holder}", key=f"btn_{holder}"):
                    st.session_state.selected_holder = holder
                    # Attempt to rerun. Some Streamlit versions may not have experimental_rerun.
                    # Use experimental_set_query_params as a fallback to trigger rerun.
                    try:
                        rerun_fn = getattr(st, "experimental_rerun", None)
                        if callable(rerun_fn):
                            rerun_fn()
                        else:
                            raise AttributeError("experimental_rerun not available")
                    except Exception:
                        # fallback trigger
                        import uuid
                        set_qp = getattr(st, "experimental_set_query_params", None)
                        if callable(set_qp):
                            set_qp(_rerun=str(uuid.uuid4()))

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
            width='stretch'
        )
    else:
        st.plotly_chart(
            portfolio_treemap(df_person),
            width='stretch'
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
        .applymap(highlight_pl_bg, subset=['P&L'])  # type: ignore[attr-defined]
        .applymap(highlight_pl_bg, subset=['P&L %'])  # type: ignore[attr-defined]
)
    st.dataframe(styled_df, width='stretch')
    # st.dataframe(style_pl_df(df_person), width='stretch')

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
    def _safe_weighted_avg(buy_series, qty_series) -> float:
        b = pd.to_numeric(buy_series, errors='coerce')
        q = pd.to_numeric(qty_series, errors='coerce')
        denom = q.sum()
        if denom == 0 or pd.isna(denom):
            return 0
        return (b * q).sum() / denom

    grouped = df_pivot.groupby('SYMBOL', as_index=False).agg({
        'Realised P/L': 'sum',
        'Unrealised P/L': 'sum',
        'Buy': lambda x: _safe_weighted_avg(x, df_pivot.loc[x.index, 'QTY']),
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

    # st.dataframe(df_display, width='stretch')
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
        # Use list arguments to satisfy type checks in some pandas stubs
        holder_change = holder_change.sort_values('Holder')  # type: ignore[call-arg]



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


        st.plotly_chart(fig_bar, width='stretch')
    else:
        st.info("No unrealised positions available for today's change.")


    styled_df = (
    df_display.style
        .format({
            'Realised P/L': '₹{:,.2f}',
            'Unrealised P/L': '₹{:,.2f}',
            'Change Today': '₹{:,.2f}'
        })
        .applymap(highlight_pl_bg, subset=['Change Today'])  # type: ignore[attr-defined]
        .applymap(color_unrealised_pl, subset=['Unrealised P/L'])  # type: ignore[attr-defined]
)

    st.dataframe(styled_df, width='stretch')




# ================= RSI ANALYSIS =================
elif page == "📉 RSI Analysis":
    st.title("📉 Interactive Stock RSI Plotter")
    st.caption("Daily, Weekly & Monthly RSI with EMA overlays")

    stock_list = get_index_symbols("NIFTY 500")
    selected_stock = st.selectbox("Select Stock", ["Select"] + stock_list)

    if selected_stock != "Select":
        ticker = selected_stock.strip().upper()
        with st.spinner("Fetching data..."):
            try:
                df = yf.download(
                    f"{ticker}.NS",
                    period='15y',
                    interval='1d',
                    auto_adjust=True
                )

                if df is None or df.empty:
                    raise ValueError("No data returned from yfinance")

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                data = df.dropna().copy()
                data['RSI'] = calculate_rsi(data['Close'])
                data['EMA_RSI'] = calculate_ema(data['RSI'])

                weekly_close = data['Close'].resample('W').last()
                weekly_ema50 = calculate_ema(weekly_close, 50)
                weekly_rsi = calculate_rsi(weekly_close)
                weekly_rsi_ema = calculate_ema(weekly_rsi, 50)

                monthly_close = data['Close'].resample('ME').last()
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
                    fig.add_hline(y=40, line_dash='dot', row=r, col=1,line_color="red")  # type: ignore[arg-type]
                    fig.add_hline(y=50, line_dash='dot', row=r, col=1)  # type: ignore[arg-type]
                    fig.add_hline(y=60, line_dash='dot', row=r, col=1,line_color="blue")  # type: ignore[arg-type]
                    fig.update_yaxes(range=[0, 100], row=r, col=1)  # type: ignore[arg-type]

                fig.update_layout(
                    height=1400,
                    hovermode='x unified',
                    template='plotly_white'
                )

                st.plotly_chart(fig, width='stretch')

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
            "Nifty LARGEMIDCAP 250",
            "Nifty 500"
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
    
    elif stock_universe == "Nifty 500":
        symbols = get_index_symbols("NIFTY 500")

    elif stock_universe == "Nifty Next 50":
        symbols = get_index_symbols("NIFTY%20NEXT%2050")

    elif stock_universe == "Nifty Midcap 150":
        symbols = get_index_symbols("NIFTY MIDCAP 150")

    elif stock_universe == "Nifty LARGEMIDCAP 250":
        symbols = get_index_symbols("NIFTY%20LARGEMID250")

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
        st.dataframe(pd.DataFrame(results), width='stretch')
    else:
        st.info("No results for selected analytics mode.")
