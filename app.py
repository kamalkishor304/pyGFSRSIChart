import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from typing import Dict, List, Optional

from ui import (
    inject_global_style,
    page_header,
    render_metric_cards,
    render_section_header,
)

from data import fetch_monthly_close, fetch_quotes, get_ltp_and_change
from analysis import calculate_rsi, calculate_ema, get_monthly_rsi_trend
from portfolio import load_all_trades, get_holdings_quotes, enrich_portfolio_df, SHEETS

st.set_page_config(page_title='Equity Portfolio Dashboard', page_icon='📈', layout='wide')

# ---- shared helpers ----

def color_unrealised_pl(val):
    try:
        val = float(str(val).replace('₹', '').replace(',', ''))
        if val < 0:
            return 'color: red; font-weight: bold'
        elif val > 0:
            return 'color: green; font-weight: bold'
    except Exception:
        pass
    return ''


def highlight_pl_bg(val):
    try:
        val = float(str(val).replace('₹', '').replace(',', ''))
        if val < 0:
            return 'background-color: #ffcccc'
        elif val > 0:
            return 'background-color: #ccffcc'
    except Exception:
        pass
    return ''


def portfolio_summary(df: pd.DataFrame):
    invested = df['Invested'].sum()
    current = df['Current'].sum()
    profit = current - invested
    return invested, current, profit


def styled_financial_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def existing_columns(columns):
        return [col for col in columns if col in df.columns]

    styled = df.style.format({
        'Buy': '₹{:,.2f}',
        'LTP': '₹{:,.2f}',
        'Invested': '₹{:,.2f}',
        'Current': '₹{:,.2f}',
        'P&L': '₹{:,.2f}',
        'P&L %': '{:.2f}%',
        'Realised P/L': '₹{:,.2f}',
        'Unrealised P/L': '₹{:,.2f}',
        'Change Today': '₹{:,.2f}',
    })
    styled = styled.applymap(color_unrealised_pl, subset=existing_columns(['P&L', 'Realised P/L', 'Unrealised P/L']))  # type: ignore[attr-defined]
    styled = styled.applymap(highlight_pl_bg, subset=existing_columns(['P&L', 'P&L %', 'Realised P/L', 'Unrealised P/L', 'Change Today']))  # type: ignore[attr-defined]
    return styled


def portfolio_pie_by_stocks(df: pd.DataFrame):
    pie_df = df.groupby('SYMBOL', as_index=False)['Current'].sum()
    fig = px.pie(
        pie_df,
        names='SYMBOL',
        values='Current',
        hole=0.58,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_traces(textinfo='percent+label', textposition='inside')
    fig.update_layout(height=420, margin=dict(t=20, b=20, l=10, r=10))
    return fig


def portfolio_treemap(df: pd.DataFrame):
    fig = px.treemap(
        df,
        path=['SYMBOL'],
        values='Current',
        color='P&L',
        color_continuous_scale='RdYlGn',
        custom_data=['Invested', 'Current', 'P&L', 'P&L %', 'QTY'],
    )
    fig.update_traces(
        texttemplate=(
            '<b>%{label}</b><br>'
            '₹%{customdata[1]:,.0f}<br>'
            'P&L: ₹%{customdata[2]:,.0f}<br>'
            '%{customdata[3]:.2f}%'
        ),
        textinfo='label+text',
    )
    fig.update_layout(height=520, margin=dict(t=20, b=20, l=10, r=10))
    return fig


def get_nse_index_symbols(index_name: str):
    import requests
    normalized_name = index_name.replace('%20', ' ')
    try:
        nse_url = f'https://www.nseindia.com/api/equity-stock-indices?index={index_name}'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(nse_url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                symbols = [item.get('symbol', '').strip() for item in data['data'] if item.get('symbol')]
                if symbols:
                    return sorted(symbols)
    except Exception:
        st.warning(f'Unable to load {normalized_name} constituents from NSE API. Using fallback list.')

    return {
        'NIFTY 50': ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ITC'],
        'NIFTY NEXT 50': ['ACC', 'AIAENG', 'AMBUJACEM'],
        'NIFTY MIDCAP 150': ['3MINDIA', 'ABCAPITAL'],
        'NIFTY LARGEMIDCAP 250': ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ITC'],
        'NIFTY 500': ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ITC'],
    }.get(normalized_name, ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ITC'])


def get_index_symbols(index_name: str):
    return get_nse_index_symbols(index_name)


# ---- load data ----
df_all = load_all_trades()
all_symbols = df_all['SYMBOL'].dropna().unique().tolist()
holders = sorted(df_all['Holder'].dropna().unique())

if 'selected_holder' not in st.session_state:
    st.session_state.selected_holder = holders[0] if holders else None


# ---- sidebar ----
with st.sidebar:
    st.markdown('# Portfolio Studio')
    st.markdown('A modern dashboard for Indian equity holdings and RSI analytics.')
    page = st.radio(
        'Navigation',
        ['Home', 'Portfolio', 'Pivot Data', 'RSI Analysis', 'Trend Analysis'],
        index=0,
    )
    st.markdown('---')
    st.caption('Built for quick portfolio decisions and clean visual storytelling.')

inject_global_style('light')

holdings_info = get_holdings_quotes(all_symbols, max_workers=12)


def safe_ltp(symbol: str):
    return holdings_info.get(symbol, (None, None))[0]


def safe_change(symbol: str):
    return holdings_info.get(symbol, (None, None))[1]


# ---- home page ----
if page == 'Home':
    page_header('Equity Portfolio Studio', 'Live holdings, allocation, performance and trend insights for your NSE portfolio.')

    with st.spinner('Refreshing dashboard...'):
        portfolio_df = enrich_portfolio_df(df_all.copy(), holdings_info)
        invested, current, profit = portfolio_summary(portfolio_df)
        profit_pct = (profit / invested * 100) if invested else 0
        total_positions = portfolio_df.shape[0]
        unique_symbols = portfolio_df['SYMBOL'].nunique()

    cards = [
        {'title': 'Total Invested', 'value': f'₹ {invested:,.0f}', 'delta': None, 'caption': f'{unique_symbols} active stocks'},
        {'title': 'Current Value', 'value': f'₹ {current:,.0f}', 'delta': f'{profit_pct:+.2f}% vs invested', 'caption': 'Real-time market value'},
        {'title': 'Portfolio P&L', 'value': f'₹ {profit:,.0f}', 'delta': None, 'caption': f'{total_positions} open positions'},
        {'title': 'Market Coverage', 'value': f'{unique_symbols} symbols', 'delta': None, 'caption': 'Current holdings universe'},
    ]
    render_metric_cards(cards)

    c1, c2 = st.columns([2, 1])
    with c1:
        render_section_header('Portfolio Allocation', 'Visualize current unrealized portfolio value by stock.')
        st.plotly_chart(portfolio_pie_by_stocks(portfolio_df), use_container_width=True)
    with c2:
        render_section_header('Top Holdings', 'Current value ranking for your largest positions.')
        top_holdings = (
            portfolio_df.groupby('SYMBOL', as_index=False)['Current']
            .sum()
            .sort_values(by='Current', ascending=False)  # type: ignore[call-overload]
            .head(8)
        )
        top_holdings['Current'] = top_holdings['Current'].map('₹{:,.0f}'.format)
        st.dataframe(top_holdings, width='stretch')

    render_section_header('Open Positions', 'Detailed view of your active equity portfolio.')
    open_positions = portfolio_df.sort_values('Current', ascending=False)
    st.dataframe(styled_financial_table(open_positions), width='stretch')

elif page == 'Portfolio':
    page_header('Holder Portfolio Detail', 'Inspect a single holder, their allocation and performance metrics.')
    selected_holder = st.selectbox('Choose a holder', holders, index=holders.index(st.session_state.selected_holder))
    st.session_state.selected_holder = selected_holder
    holder_df = df_all[df_all['Holder'] == selected_holder]

    with st.spinner(f'Loading portfolio for {selected_holder}...'):
        holder_df = enrich_portfolio_df(holder_df, holdings_info)
        invested, current, profit = portfolio_summary(holder_df)
        profit_pct = (profit / invested * 100) if invested else 0

    cards = [
        {'title': 'Invested', 'value': f'₹ {invested:,.0f}', 'delta': None, 'caption': selected_holder},
        {'title': 'Market Value', 'value': f'₹ {current:,.0f}', 'delta': f'{profit_pct:+.2f}% vs invested', 'caption': 'Current position value'},
        {'title': 'Net P&L', 'value': f'₹ {profit:,.0f}', 'delta': None, 'caption': 'Unrealised profit and loss'},
        {'title': 'Stock Count', 'value': f'{holder_df.shape[0]}', 'delta': None, 'caption': 'Open positions'},
    ]
    render_metric_cards(cards)

    view = st.radio('Chart view', ['Allocation', 'Trend'], horizontal=True)
    if view == 'Allocation':
        st.plotly_chart(portfolio_treemap(holder_df), use_container_width=True)
    else:
        st.plotly_chart(portfolio_pie_by_stocks(holder_df), use_container_width=True)

    render_section_header('Position Details', 'Review buy dates, LTP, invested amount and unrealized P&L.')
    holder_df_display = holder_df.assign(**{
        'Buy Date': holder_df['Buy Date'].dt.strftime('%Y-%m-%d'),
    })
    st.dataframe(styled_financial_table(holder_df_display), width='stretch')

elif page == 'Pivot Data':
    page_header('Pivot Data Explorer', 'Compare holdings and change summaries across the full dataset.')
    selected_holder = st.selectbox('Select holder summary', ['All Holders'] + holders)
    pivot_df = df_all.copy() if selected_holder == 'All Holders' else df_all[df_all['Holder'] == selected_holder]
    pivot_df['Change'] = pivot_df['SYMBOL'].apply(safe_change)
    pivot_df['LTP'] = pivot_df['SYMBOL'].apply(safe_ltp)
    pivot_df['Realised P/L'] = pivot_df.apply(
        lambda row: (row['Sell'] - row['Buy']) * row['QTY'] if pd.notna(row['Sell']) and row['Sell'] > 0 else 0,
        axis=1,
    )
    pivot_df['Unrealised P/L'] = pivot_df.apply(
        lambda row: (row['LTP'] - row['Buy']) * row['QTY'] if pd.isna(row['Sell']) or row['Sell'] == 0 else 0,
        axis=1,
    )

    def safe_avg(series, weights):
        values = pd.to_numeric(series, errors='coerce')
        total_weights = pd.to_numeric(weights, errors='coerce').sum()
        return (values * weights).sum() / total_weights if total_weights else 0

    grouped = pivot_df.groupby('SYMBOL', as_index=False).agg({
        'Realised P/L': 'sum',
        'Unrealised P/L': 'sum',
        'Buy': lambda x: safe_avg(x, pivot_df.loc[x.index, 'QTY']),
        'QTY': 'sum',
        'Change': 'last',
    })
    grouped['Change Today'] = grouped.apply(
        lambda row: 0 if row['Unrealised P/L'] == 0 else row['Change'] * row['QTY'],
        axis=1,
    )

    render_section_header('Daily Change Summary', 'Unrealised change in ₹ across active positions.')
    change_df = pivot_df[pivot_df['Unrealised P/L'] != 0].copy()
    if not change_df.empty:
        change_df['Today Change ₹'] = change_df['Change'] * change_df['QTY']
        holder_change = (
            change_df.groupby('Holder', as_index=False)['Today Change ₹'].sum()
            .assign(Direction=lambda df: df['Today Change ₹'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Zero'))
            .sort_values('Holder')
        )
        fig = px.bar(
            holder_change,
            x='Holder',
            y='Today Change ₹',
            color='Direction',
            text='Today Change ₹',
            color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Zero': 'gray'},
        )
        fig.update_traces(texttemplate='₹ %{text:,.2f}', textposition='inside')
        fig.update_layout(height=460, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No unrealised positions available for today.')

    render_section_header('Pivot Table', 'See the aggregated holding analytics for each symbol.')
    st.dataframe(styled_financial_table(grouped), width='stretch')

elif page == 'RSI Analysis':
    page_header('RSI Analysis', 'Explore RSI momentum with daily, weekly and monthly signals.')
    stock_list = get_index_symbols('NIFTY 500')
    selected_stock = st.selectbox('Select stock for RSI analysis', ['Select'] + stock_list)
    if selected_stock != 'Select':
        with st.spinner('Building RSI chart...'):
            ticker = selected_stock.strip().upper()
            try:
                df = yf.download(f'{ticker}.NS', period='15y', interval='1d', auto_adjust=True)
                if df is None or df.empty:
                    raise ValueError('No data returned from yfinance')
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
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    row_heights=[0.35, 0.2, 0.2, 0.2],
                    subplot_titles=[
                        f'{ticker} Weekly Close Price',
                        'Monthly RSI',
                        'Weekly RSI',
                        'Daily RSI',
                    ],
                )
                fig.add_trace(go.Scatter(x=weekly_close.index, y=weekly_close, name='Weekly Close', line=dict(width=3)), row=1, col=1)
                fig.add_trace(go.Scatter(x=weekly_ema50.index, y=weekly_ema50, name='50 EMA', line=dict(width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=monthly_rsi.index, y=monthly_rsi, name='Monthly RSI'), row=2, col=1)
                fig.add_trace(go.Scatter(x=monthly_rsi_ema.index, y=monthly_rsi_ema, name='EMA RSI'), row=2, col=1)
                fig.add_trace(go.Scatter(x=weekly_rsi.index, y=weekly_rsi, name='Weekly RSI'), row=3, col=1)
                fig.add_trace(go.Scatter(x=weekly_rsi_ema.index, y=weekly_rsi_ema, name='EMA RSI'), row=3, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['EMA_RSI'], name='Daily RSI EMA'), row=4, col=1)

                for row_id in [2, 3, 4]:
                    fig.add_hline(y=40, line_dash='dot', row=row_id, col=1, line_color='red')  # type: ignore[arg-type]
                    fig.add_hline(y=50, line_dash='dot', row=row_id, col=1)  # type: ignore[arg-type]
                    fig.add_hline(y=60, line_dash='dot', row=row_id, col=1, line_color='blue')  # type: ignore[arg-type]
                    fig.update_yaxes(range=[0, 100], row=row_id, col=1)  # type: ignore[arg-type]

                fig.update_layout(height=1400, hovermode='x unified', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f'Unable to load RSI data for {ticker}: {exc}')

elif page == 'Trend Analysis':
    page_header('Trend Analysis', 'Monthly RSI trend strength, momentum slope, and flip detection.')
    analysis_mode = st.selectbox('Analysis mode', ['Trend Strength (Strong / Weak)', 'RSI Slope (Momentum)', 'Trend Flip Detection'])
    stock_universe = st.selectbox('Analyze stocks from', ['Current Holding', 'Nifty 50', 'Nifty Next 50', 'Nifty Midcap 150', 'Nifty LARGEMIDCAP 250', 'Nifty 500'])

    if stock_universe == 'Current Holding':
        symbols = sorted(df_all.loc[df_all['Sell'] == 0, 'SYMBOL'].dropna().unique())
    elif stock_universe == 'Nifty 50':
        symbols = get_index_symbols('NIFTY 50')
    elif stock_universe == 'Nifty 500':
        symbols = get_index_symbols('NIFTY 500')
    elif stock_universe == 'Nifty Next 50':
        symbols = get_index_symbols('NIFTY%20NEXT%2050')
    elif stock_universe == 'Nifty Midcap 150':
        symbols = get_index_symbols('NIFTY MIDCAP 150')
    else:
        symbols = get_index_symbols('NIFTY%20LARGEMID250')

    trend_up, trend_down, slope_improving, slope_weakening, results = [], [], [], [], []
    progress = st.progress(0)
    status = st.empty()

    def analyze_symbol(symbol: str):
        data = get_monthly_rsi_trend(symbol)
        if not data:
            return None
        rsi = data['Monthly RSI']
        ema = data['Monthly RSI EMA']
        prev_rsi = data['Prev RSI']
        prev_ema = data['Prev EMA']
        row = []
        if analysis_mode == 'Trend Strength (Strong / Weak)':
            if rsi >= 60 and rsi > ema:
                trend = '🟢 Strong Uptrend'
            elif rsi >= 50 and rsi < ema:
                trend = '🟡 Weak Uptrend'
            elif rsi >= 40 and rsi < ema:
                trend = '🟠 Weak Downtrend'
            elif rsi < 40 and rsi < ema:
                trend = '🔴 Strong Downtrend'
            else:
                trend = '⚪ Neutral'
            row = [{'SYMBOL': symbol, 'RSI': rsi, 'Trend': trend}]
        elif analysis_mode == 'RSI Slope (Momentum)':
            slope = round(rsi - prev_rsi, 2)
            momentum = '🟢 Improving' if slope > 0 else '🔴 Weakening'
            row = [{'SYMBOL': symbol, 'RSI Slope': slope, 'Momentum': momentum}]
        else:
            prev_trend = 'Uptrend' if prev_rsi > prev_ema else 'Downtrend'
            curr_trend = 'Uptrend' if rsi > ema else 'Downtrend'
            if prev_trend != curr_trend:
                flip = '🚀 Bullish' if curr_trend == 'Uptrend' else '⚠️ Bearish'
                row = [{'SYMBOL': symbol, 'Previous': prev_trend, 'Current': curr_trend, 'Flip': flip}]

        if rsi >= 50:
            trend_up.append({'SYMBOL': symbol, 'Monthly RSI': rsi})
        else:
            trend_down.append({'SYMBOL': symbol, 'Monthly RSI': rsi})
        slope_value = round(rsi - prev_rsi, 2)
        if slope_value > 0:
            slope_improving.append({'SYMBOL': symbol, 'RSI Slope': slope_value})
        elif slope_value < 0:
            slope_weakening.append({'SYMBOL': symbol, 'RSI Slope': slope_value})
        return row

    for index, symbol in enumerate(symbols):
        status.text(f'Analyzing {symbol} ({index + 1}/{len(symbols)})')
        result = analyze_symbol(symbol)
        if result:
            results.extend(result)
        progress.progress((index + 1) / len(symbols))

    progress.empty()
    status.empty()

    render_metric_cards([
        {'title': 'Stocks Evaluated', 'value': f'{len(symbols)}', 'delta': None, 'caption': stock_universe},
        {'title': 'RSI Bullish', 'value': f'{len(trend_up)}', 'delta': None, 'caption': 'Monthly RSI >= 50'},
        {'title': 'RSI Bearish', 'value': f'{len(trend_down)}', 'delta': None, 'caption': 'Monthly RSI < 50'},
        {'title': 'Momentum shifts', 'value': f'{len(slope_improving) + len(slope_weakening)}', 'delta': None, 'caption': 'Improving and weakening slopes'},
    ])

    if results:
        st.dataframe(pd.DataFrame(results).sort_values(by='SYMBOL'), width='stretch')
    else:
        st.info('No trend results for the selected universe.')
