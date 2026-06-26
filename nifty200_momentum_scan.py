import datetime as dt
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_NIFTY_200_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'LT.NS',
    'KOTAKBANK.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'MARUTI.NS',
    'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'NTPC.NS', 'POWERGRID.NS',
    'ASIANPAINT.NS', 'M&M.NS', 'HEROMOTOCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS',
    'WIPRO.NS', 'DRREDDY.NS', 'ADANIPORTS.NS', 'CIPLA.NS', 'DIVISLAB.NS',
    'HCLTECH.NS', 'EICHERMOT.NS', 'INDUSINDBK.NS', 'NESTLEIND.NS', 'BRITANNIA.NS',
    'GRASIM.NS', 'TECHM.NS', 'HDFCLIFE.NS', 'UPL.NS', 'SBILIFE.NS',
    'TATAMOTORS.NS', 'TATASTEEL.NS', 'PIDILITIND.NS', 'SHREECEM.NS', 'SIEMENS.NS',
    'MUTHOOTFIN.NS', 'TRENT.NS', 'DABUR.NS', 'TVSMOTOR.NS', 'PFC.NS', 'REC.NS',
    'BEL.NS', 'ONGC.NS', 'GAIL.NS', 'IOC.NS', 'BPCL.NS', 'MRF.NS', 'CHOLAFIN.NS',
    'LICI.NS', 'HAVELLS.NS', 'ABBOTINDIA.NS', 'AUBANK.NS', 'APOLLOHOSP.NS',
    'DMART.NS', 'INDIGO.NS', 'DLF.NS', 'PNB.NS', 'BANKBARODA.NS', 'CANBK.NS',
    'IDFCFIRSTB.NS', 'IRCTC.NS', 'JINDALSTEL.NS', 'LUPIN.NS', 'MCDOWELL-N.NS',
    'NALCO.NS', 'PAGEIND.NS', 'PETRONET.NS', 'POLYCAB.NS', 'SRF.NS', 'TATAPOWER.NS',
    'VOLTAS.NS', 'AMBUJACEM.NS', 'BANDHANBNK.NS', 'COLPAL.NS', 'FEDERALBNK.NS',
    'GODREJCP.NS', 'JUBLFOOD.NS', 'METROPOLIS.NS', 'MOTHERSON.NS', 'PAYTM.NS',
    'PIIND.NS', 'RAMCOCEM.NS', 'SYNGENE.NS', 'UBL.NS', 'UNITDSPR.NS', 'ZEEL.NS'
]


def _normalize_end_date(end_date: Optional[dt.date]) -> str:
    if end_date is None:
        return dt.date.today().strftime('%Y-%m-%d')
    if isinstance(end_date, dt.datetime):
        return end_date.date().strftime('%Y-%m-%d')
    return end_date.strftime('%Y-%m-%d')


def _prepare_data(ticker: str, start_date: str, end_date: Optional[dt.date]) -> Optional[pd.DataFrame]:
    end_date_str = _normalize_end_date(end_date)
    try:
        df = yf.download(ticker, start=start_date, end=end_date_str, auto_adjust=True, progress=False)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df.columns = [col.lower() for col in df.columns]

    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        return None

    df = df[required].dropna()
    if len(df) < 252:
        return None
    return df


def _zscore(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=values.index)
    return (values - mean) / std


def _detect_recent_breakout(
    df: pd.DataFrame,
    breakout_window: int = 20,
    max_age_days: int = 2,
    min_volume_spike: float = 1.5,
    max_proximity_pct: float = 3.0,
) -> Optional[dict]:
    if len(df) <= breakout_window:
        return None

    prev_high = df['high'].shift(1).rolling(breakout_window, min_periods=breakout_window).max()
    prev_vol_avg = df['volume'].shift(1).rolling(breakout_window, min_periods=breakout_window).mean()
    breakout_signal = (
        (df['close'] > prev_high)
        & (df['volume'] > prev_vol_avg * min_volume_spike)
        & (df['close'] > df['close'].shift(1))
    )

    if not breakout_signal.any():
        return None

    last_idx = int(np.flatnonzero(breakout_signal.to_numpy())[-1])
    breakout_level = float(prev_high.iloc[last_idx])
    breakout_volume = float(df['volume'].iloc[last_idx])
    avg_volume_before_breakout = float(prev_vol_avg.iloc[last_idx])
    volume_spike = breakout_volume / avg_volume_before_breakout if avg_volume_before_breakout > 0 else np.nan
    breakout_age_days = int((df.index[-1] - df.index[last_idx]).days)
    distance_to_breakout_pct = float((df['close'].iloc[-1] / breakout_level - 1.0) * 100.0) if breakout_level else np.nan

    if breakout_age_days > max_age_days:
        return None

    if np.isnan(distance_to_breakout_pct) or abs(distance_to_breakout_pct) > max_proximity_pct:
        return None

    if np.isnan(volume_spike) or volume_spike < min_volume_spike:
        return None

    return {
        'breakout_age_days': breakout_age_days,
        'breakout_level': breakout_level,
        'distance_to_breakout_pct': distance_to_breakout_pct,
        'volume_spike': volume_spike,
        'breakout_day': df.index[last_idx],
    }


def calculate_momentum_scores(
    tickers: Optional[List[str]] = None,
    start_date: str = '2020-01-01',
    end_date: Optional[dt.date] = None,
    min_price: float = 50.0,
    min_volume: float = 1_000_000.0,
) -> pd.DataFrame:
    ticker_list = tickers or DEFAULT_NIFTY_200_TICKERS
    rows = []

    for ticker in ticker_list:
        df = _prepare_data(ticker, start_date, end_date)
        if df is None:
            continue

        if df['close'].iloc[-1] < min_price:
            continue

        avg_volume = df['volume'].tail(20).mean()
        if avg_volume * df['close'].iloc[-1] < min_volume:
            continue

        daily_returns = df['close'].pct_change().dropna()
        ret_6m = df['close'].iloc[-1] / df['close'].shift(126).iloc[-1] - 1.0 if len(df) > 126 else np.nan
        ret_12m = df['close'].iloc[-1] / df['close'].shift(252).iloc[-1] - 1.0 if len(df) > 252 else np.nan
        vol_6m = daily_returns.tail(126).std() * np.sqrt(252) if len(daily_returns) >= 126 else np.nan
        vol_12m = daily_returns.tail(252).std() * np.sqrt(252) if len(daily_returns) >= 252 else np.nan

        if not np.isfinite(ret_6m) or not np.isfinite(ret_12m) or not np.isfinite(vol_6m) or not np.isfinite(vol_12m):
            continue

        score_6m = ret_6m / vol_6m if vol_6m > 0 else np.nan
        score_12m = ret_12m / vol_12m if vol_12m > 0 else np.nan
        rows.append({
            'ticker': ticker,
            'close': float(df['close'].iloc[-1]),
            'ret_6m': float(ret_6m),
            'ret_12m': float(ret_12m),
            'score_6m': float(score_6m),
            'score_12m': float(score_12m),
            'avg_volume': float(avg_volume),
        })

    if not rows:
        return pd.DataFrame(columns=['ticker', 'close', 'ret_6m', 'ret_12m', 'score_6m', 'score_12m', 'avg_volume'])

    df_scores = pd.DataFrame(rows)
    df_scores['z_score_6m'] = _zscore(df_scores['score_6m'])
    df_scores['z_score_12m'] = _zscore(df_scores['score_12m'])
    df_scores['composite_score'] = (df_scores['z_score_6m'] + df_scores['z_score_12m']) / 2.0
    df_scores = df_scores.sort_values('composite_score', ascending=False).reset_index(drop=True)
    return df_scores


def build_chartink_scan_logic(max_price: float = 800.0) -> str:
    return (
        '({nifty200}) AND (latest close > latest sma(close, 50)) '
        'AND (latest close > latest sma(close, 200)) '
        'AND (latest close > latest highest(high, 20)) '
        'AND (latest return(close, 126) > 0) '
        'AND (latest return(close, 252) > 0) '
        'AND (latest volume > latest sma(volume, 20) * 1.5) '
        'AND (latest close > 50) '
        f'AND (latest close < {max_price})'
    )


def screen_recent_breakouts(
    tickers: Optional[List[str]] = None,
    start_date: str = '2020-01-01',
    end_date: Optional[dt.date] = None,
    top_n: int = 30,
    min_price: float = 50.0,
    max_price: float = 800.0,
    min_volume: float = 1_000_000.0,
) -> pd.DataFrame:
    ticker_list = tickers or DEFAULT_NIFTY_200_TICKERS
    rows = []

    for ticker in ticker_list:
        df = _prepare_data(ticker, start_date, end_date)
        if df is None:
            continue

        if df['close'].iloc[-1] < min_price or df['close'].iloc[-1] >= max_price:
            continue

        avg_volume = df['volume'].tail(20).mean()
        if avg_volume * df['close'].iloc[-1] < min_volume:
            continue

        breakout_info = _detect_recent_breakout(df)
        if breakout_info is None:
            continue

        close = float(df['close'].iloc[-1])
        daily_returns = df['close'].pct_change().dropna()
        ret_6m = df['close'].iloc[-1] / df['close'].shift(126).iloc[-1] - 1.0 if len(df) > 126 else np.nan
        ret_12m = df['close'].iloc[-1] / df['close'].shift(252).iloc[-1] - 1.0 if len(df) > 252 else np.nan
        vol_6m = daily_returns.tail(126).std() * np.sqrt(252) if len(daily_returns) >= 126 else np.nan
        vol_12m = daily_returns.tail(252).std() * np.sqrt(252) if len(daily_returns) >= 252 else np.nan
        score_6m = ret_6m / vol_6m if vol_6m > 0 else np.nan
        score_12m = ret_12m / vol_12m if vol_12m > 0 else np.nan

        rows.append({
            'ticker': ticker,
            'close': close,
            'ret_6m': float(ret_6m),
            'ret_12m': float(ret_12m),
            'score_6m': float(score_6m),
            'score_12m': float(score_12m),
            'avg_volume': float(avg_volume),
            'breakout_age_days': breakout_info['breakout_age_days'],
            'breakout_level': breakout_info['breakout_level'],
            'distance_to_breakout_pct': breakout_info['distance_to_breakout_pct'],
            'volume_spike': breakout_info['volume_spike'],
            'breakout_day': breakout_info['breakout_day'],
        })

    if not rows:
        return pd.DataFrame(columns=[
            'ticker', 'close', 'ret_6m', 'ret_12m', 'score_6m', 'score_12m', 'avg_volume',
            'breakout_age_days', 'breakout_level', 'distance_to_breakout_pct', 'volume_spike', 'breakout_day'
        ])

    df_breakouts = pd.DataFrame(rows)
    df_breakouts = df_breakouts.sort_values(
        ['breakout_age_days', 'distance_to_breakout_pct', 'volume_spike'],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return df_breakouts.head(top_n)


def scan_nifty200_momentum_30(
    tickers: Optional[List[str]] = None,
    start_date: str = '2020-01-01',
    end_date: Optional[dt.date] = None,
    top_n: int = 30,
    min_price: float = 50.0,
    max_price: float = 800.0,
    min_volume: float = 1_000_000.0,
) -> pd.DataFrame:
    return screen_recent_breakouts(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
        min_price=min_price,
        max_price=max_price,
        min_volume=min_volume,
    )


def main() -> None:
    print('NIFTY 200 Momentum 30 style scan (rule-based swing proxy)')
    print('Chartink-style logic:')
    print(build_chartink_scan_logic(max_price=800.0))
    print('\nRecent breakout watchlist for sub-₹800 stocks (1-2 days back, near breakout, strong volume):')
    ranked = screen_recent_breakouts(top_n=15, end_date=None, max_price=800.0)
    if ranked.empty:
        print('No candidates found.')
        return

    for _, row in ranked.iterrows():
        print(
            f"{row['ticker']}: close={row['close']:.2f}, breakout_age={row['breakout_age_days']}d, "
            f"distance={row['distance_to_breakout_pct']:.2f}%, volume_spike={row['volume_spike']:.2f}x"
        )


if __name__ == '__main__':
    main()
