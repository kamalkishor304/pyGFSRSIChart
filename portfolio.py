import pandas as pd
from data import fetch_quotes


SHEETS = [
    "Aishwarya",
    "Sudhadevi",
    "Ramkishor",
    "Rashmi",
    "Dashrathlal"
]


def load_all_trades(path="Equity Trades.xlsx") -> pd.DataFrame:
    dfs = []
    for sheet in SHEETS:
        df = pd.read_excel(path, sheet_name=sheet)
        df['Buy Date'] = pd.to_datetime(df['Buy Date'], errors='coerce')
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


def get_holdings_quotes(symbols, max_workers=8):
    return fetch_quotes(symbols, max_workers=max_workers)


def enrich_portfolio_df(df: pd.DataFrame, holdings_info: dict) -> pd.DataFrame:
    df = df.copy()
    df = df[(df['Sell'].isna()) | (df['Sell'] == 0)]
    df['LTP'] = df['SYMBOL'].apply(lambda s: holdings_info.get(s, (None, None))[0])
    df['Invested'] = df['Buy'] * df['QTY']
    df['Current'] = df['LTP'] * df['QTY']
    df['P&L'] = df['Current'] - df['Invested']
    df['P&L %'] = (df['P&L'] / df['Invested']) * 100
    return df
