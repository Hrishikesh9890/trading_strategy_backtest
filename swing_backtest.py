import datetime
from datetime import date, datetime as dt
from typing import List, Optional, Union

import backtrader as bt
import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_NIFTY_100_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'LT.NS',
    'KOTAKBANK.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'MARUTI.NS',
    'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'NTPC.NS', 'POWERGRID.NS',
    'ASIANPAINT.NS', 'M&M.NS', 'HEROMOTOCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS',
    'WIPRO.NS', 'DRREDDY.NS', 'ADANIPORTS.NS', 'CIPLA.NS', 'DIVISLAB.NS',
    'HCLTECH.NS', 'EICHERMOT.NS', 'INDUSINDBK.NS', 'NESTLEIND.NS', 'BRITANNIA.NS',
    'GRASIM.NS', 'TECHM.NS', 'HDFCLIFE.NS', 'UPL.NS', 'SBILIFE.NS',
    'TATAMOTORS.NS', 'TATASTEEL.NS', 'PIDILITIND.NS', 'SHREECEM.NS', 'SIEMENS.NS',
    'MUTHOOTFIN.NS', 'ZOMATO.NS', 'TRENT.NS', 'DABUR.NS', 'TVSMOTOR.NS',
    'PFC.NS', 'REC.NS', 'BEL.NS', 'ONGC.NS', 'GAIL.NS', 'IOC.NS', 'BPCL.NS',
    'MRF.NS', 'CHOLAFIN.NS', 'LICI.NS', 'HAVELLS.NS', 'ABBOTINDIA.NS',
    'AUBANK.NS', 'APOLLOHOSP.NS', 'DMART.NS', 'INDIGO.NS', 'DLF.NS', 'PNB.NS',
    'BANKBARODA.NS', 'CANBK.NS', 'IDFCFIRSTB.NS', 'IRCTC.NS', 'JINDALSTEL.NS',
    'LUPIN.NS', 'MCDOWELL-N.NS', 'NALCO.NS', 'PAGEIND.NS', 'PETRONET.NS',
    'POLYCAB.NS', 'SRF.NS', 'TATAPOWER.NS', 'VOLTAS.NS', 'WIPRO.NS'
]
DEFAULT_NIFTY_200_TICKERS = DEFAULT_NIFTY_100_TICKERS


class MotherCandleStrategy(bt.Strategy):
    params = (
        ('target_profit_pct', 0.05),
        ('max_hold_days', 5),
        ('body_avg_period', 20),
        ('volume_avg_period', 20),
        ('body_multiplier', 2.2),
        ('volume_multiplier', 1.8),
        ('verbose', False),
    )

    def __init__(self):
        self.order = None
        self.signal_day_high = None
        self.signal_day_low = None
        self.entry_price = None
        self.holding_days = 0
        self.body_avg = bt.indicators.SimpleMovingAverage(abs(self.data.close - self.data.open), period=self.params.body_avg_period)
        self.volume_avg = bt.indicators.SimpleMovingAverage(self.data.volume, period=self.params.volume_avg_period)

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.signal_day_low is not None and self.data.close[0] < self.signal_day_low:
                self.log(f'SIGNAL INVALIDATED: Price {self.data.close[0]:.2f} below signal-day low {self.signal_day_low:.2f}')
                self.signal_day_high = None
                self.signal_day_low = None
                return

            if self.signal_day_high is None:
                body = abs(self.data.close[0] - self.data.open[0])
                avg_body = self.body_avg[0]
                avg_volume = self.volume_avg[0]
                if avg_body and avg_volume and body > (avg_body * self.params.body_multiplier) and self.data.volume[0] > (avg_volume * self.params.volume_multiplier):
                    self.signal_day_high = self.data.high[0]
                    self.signal_day_low = self.data.low[0]
                    self.log(
                        f'MOTHER CANDLE: body={body:.2f}, avg_body={avg_body:.2f}, '
                        f'vol={self.data.volume[0]:.0f}, avg_vol={avg_volume:.0f}'
                    )

            if self.signal_day_high is not None and self.data.close[0] > self.signal_day_high:
                self.log(f'BUY CREATE, Price: {self.data.close[0]:.2f}, Signal High: {self.signal_day_high:.2f}')
                self.order = self.buy()
        else:
            if self.data.close[0] < self.signal_day_low:
                self.log(f'SELL (STOP LOSS) CREATE, Price: {self.data.close[0]:.2f}, SL: {self.signal_day_low:.2f}')
                self.order = self.sell()
            elif self.data.close[0] >= self.entry_price * (1 + self.params.target_profit_pct):
                self.log(f'SELL (TAKE PROFIT) CREATE, Price: {self.data.close[0]:.2f}, Entry: {self.entry_price:.2f}')
                self.order = self.sell()
            else:
                self.holding_days += 1
                if self.holding_days >= self.params.max_hold_days:
                    self.log(f'SELL (TIME EXIT) CREATE, Price: {self.data.close[0]:.2f}, Held for {self.holding_days} days')
                    self.order = self.sell()

    def log(self, txt, dt=None):
        if not self.params.verbose:
            return
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
                self.entry_price = order.executed.price
                self.holding_days = 0
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
                self.signal_day_high = None
                self.signal_day_low = None
                self.entry_price = None
                self.holding_days = 0
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            self.signal_day_high = None
            self.signal_day_low = None
            self.entry_price = None
            self.holding_days = 0

        self.order = None


class DeliverySwingStrategy(bt.Strategy):
    params = (
        ('target_profit_pct', 0.05),
        ('max_hold_days', 5),
        ('volume_avg_period', 30),
        ('volume_spike_factor', 2.0),
        ('verbose', False),
    )

    def __init__(self):
        self.order = None
        self.signal_day_high = None
        self.signal_day_low = None
        self.entry_price = None
        self.holding_days = 0
        self.delivery_proxy = None
        self.volume_avg = bt.indicators.SimpleMovingAverage(self.data.volume, period=self.params.volume_avg_period)

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.signal_day_low is not None and self.data.close[0] < self.signal_day_low:
                self.log(f'SIGNAL INVALIDATED: Price {self.data.close[0]:.2f} below signal-day low {self.signal_day_low:.2f}')
                self.signal_day_high = None
                self.signal_day_low = None
                return

            if self.signal_day_high is None:
                avg_volume = self.volume_avg[0]
                if avg_volume and self.data.volume[0] > (avg_volume * self.params.volume_spike_factor):
                    self.delivery_proxy = self.data.volume[0] / avg_volume
                    self.signal_day_high = self.data.high[0]
                    self.signal_day_low = self.data.low[0]
                    self.log(
                        f'ACCUMULATION SIGNAL: volume spike {self.delivery_proxy:.2f}x avg, '
                        f'high={self.signal_day_high:.2f}, low={self.signal_day_low:.2f}'
                    )

            if self.signal_day_high is not None and self.data.close[0] > self.signal_day_high:
                self.log(f'BUY CREATE, Price: {self.data.close[0]:.2f}, Signal High: {self.signal_day_high:.2f}')
                self.order = self.buy()
        else:
            if self.data.close[0] < self.signal_day_low:
                self.log(f'SELL (STOP LOSS) CREATE, Price: {self.data.close[0]:.2f}, SL: {self.signal_day_low:.2f}')
                self.order = self.sell()
            elif self.data.close[0] >= self.entry_price * (1 + self.params.target_profit_pct):
                self.log(f'SELL (TAKE PROFIT) CREATE, Price: {self.data.close[0]:.2f}, Entry: {self.entry_price:.2f}')
                self.order = self.sell()
            else:
                self.holding_days += 1
                if self.holding_days >= self.params.max_hold_days:
                    self.log(f'SELL (TIME EXIT) CREATE, Price: {self.data.close[0]:.2f}, Held for {self.holding_days} days')
                    self.order = self.sell()

    def log(self, txt, dt=None):
        if not self.params.verbose:
            return
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
                self.entry_price = order.executed.price
                self.holding_days = 0
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
                self.signal_day_high = None
                self.signal_day_low = None
                self.entry_price = None
                self.holding_days = 0
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            self.signal_day_high = None
            self.signal_day_low = None
            self.entry_price = None
            self.holding_days = 0

        self.order = None


def _normalize_end_date(end_date: Optional[Union[str, date, dt]]) -> str:
    if end_date is None:
        return date.today().strftime('%Y-%m-%d')
    if isinstance(end_date, dt):
        return end_date.date().strftime('%Y-%m-%d')
    if isinstance(end_date, date):
        return end_date.strftime('%Y-%m-%d')
    return str(end_date)


def _prepare_data(ticker: str, start_date: str, end_date: Optional[Union[str, date, dt]]) -> Optional[pd.DataFrame]:
    end_date = _normalize_end_date(end_date)
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
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

    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        return None

    df = df[required_columns].dropna()
    if len(df) < 60:
        return None
    return df


def _run_backtest(
    ticker: str,
    start_date: str,
    end_date: Optional[Union[str, date, dt]],
    strategy_cls: type,
    cash: float,
    commission: float,
    plot: bool,
    verbose: bool,
) -> Optional[dict]:
    end_date = _normalize_end_date(end_date)
    df_data = _prepare_data(ticker, start_date, end_date)
    if df_data is None:
        return None

    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_cls, verbose=verbose)

    data = bt.feeds.PandasData(
        dataname=df_data,
        fromdate=datetime.datetime.strptime(start_date, '%Y-%m-%d'),
        todate=datetime.datetime.strptime(end_date, '%Y-%m-%d'),
    )
    cerebro.adddata(data)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')

    result = cerebro.run()
    strategy = result[0]

    trade_analysis = strategy.analyzers.tradeanalyzer.get_analysis()
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    winning_trades = trade_analysis.get('won', {}).get('total', 0)
    losing_trades = trade_analysis.get('lost', {}).get('total', 0)
    sharpe_ratio = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
    max_drawdown = strategy.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0.0)
    final_value = cerebro.broker.getvalue()
    net_return_pct = ((final_value / cash) - 1.0) * 100.0
    accuracy_pct = (winning_trades / total_trades * 100.0) if total_trades else 0.0

    if plot and total_trades > 0:
        cerebro.plot(style='candlestick')

    return {
        'ticker': ticker,
        'final_value': final_value,
        'net_return_pct': net_return_pct,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'accuracy_pct': accuracy_pct,
    }


def run_single_backtest(
    ticker: str,
    start_date: str = '2020-01-01',
    end_date: Optional[Union[str, date, dt]] = None,
    cash: float = 100000.0,
    commission: float = 0.001,
    plot: bool = False,
    verbose: bool = False,
) -> Optional[dict]:
    return _run_backtest(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        strategy_cls=DeliverySwingStrategy,
        cash=cash,
        commission=commission,
        plot=plot,
        verbose=verbose,
    )


def run_single_mother_candle_backtest(
    ticker: str,
    start_date: str = '2020-01-01',
    end_date: Optional[Union[str, date, dt]] = None,
    cash: float = 100000.0,
    commission: float = 0.001,
    plot: bool = False,
    verbose: bool = False,
) -> Optional[dict]:
    return _run_backtest(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        strategy_cls=MotherCandleStrategy,
        cash=cash,
        commission=commission,
        plot=plot,
        verbose=verbose,
    )


def _get_current_mother_candle_status(df: pd.DataFrame) -> Optional[dict]:
    if df is None or df.empty or len(df) < 25:
        return None

    df = df.copy()
    body = (df['close'] - df['open']).abs()
    avg_body = body.rolling(20).mean()
    avg_volume = df['volume'].rolling(20).mean()
    body_ratio = body / avg_body.replace(0, np.nan)
    volume_ratio = df['volume'] / avg_volume.replace(0, np.nan)

    for idx in range(len(df) - 1, -1, -1):
        row = df.iloc[idx]
        if pd.isna(avg_body.iloc[idx]) or pd.isna(avg_volume.iloc[idx]):
            continue
        if body.iloc[idx] > (avg_body.iloc[idx] * 2.2) and df['volume'].iloc[idx] > (avg_volume.iloc[idx] * 1.8):
            signal_high = row['high']
            signal_low = row['low']
            status = 'forming'
            if idx < len(df) - 1:
                next_close = df.iloc[idx + 1]['close']
                if next_close > signal_high:
                    status = 'formed'
            return {
                'date': df.index[idx].strftime('%Y-%m-%d'),
                'status': status,
                'high': float(signal_high),
                'low': float(signal_low),
                'body_ratio': float(body_ratio.iloc[idx]),
                'volume_ratio': float(volume_ratio.iloc[idx]),
            }
    return None


def scan_mother_candle_candidates(
    tickers: Optional[List[str]] = None,
    start_date: str = '2020-01-01',
    end_date: Optional[Union[str, date, dt]] = None,
    max_candidates: int = 10,
) -> List[dict]:
    ticker_list = tickers or DEFAULT_NIFTY_100_TICKERS
    results = []

    for ticker in ticker_list:
        try:
            metrics = run_single_mother_candle_backtest(ticker, start_date=start_date, end_date=end_date)
        except Exception as exc:
            print(f'Skipping {ticker}: {exc}')
            continue

        if metrics and metrics['total_trades'] > 0:
            results.append(metrics)

    results.sort(
        key=lambda item: (
            item['net_return_pct'],
            item['accuracy_pct'],
            item['sharpe_ratio'] if item['sharpe_ratio'] is not None else float('-inf'),
        ),
        reverse=True,
    )
    return results[:max_candidates]


def scan_current_mother_candle_candidates(
    tickers: Optional[List[str]] = None,
    start_date: str = '2020-01-01',
    end_date: Optional[Union[str, date, dt]] = None,
    max_candidates: int = 10,
) -> List[dict]:
    ticker_list = tickers or DEFAULT_NIFTY_100_TICKERS
    results = []

    for ticker in ticker_list:
        try:
            df = _prepare_data(ticker, start_date, end_date)
        except Exception:
            continue
        if df is None:
            continue
        status = _get_current_mother_candle_status(df)
        if status:
            results.append({'ticker': ticker, **status})

    results.sort(key=lambda item: (item['body_ratio'], item['volume_ratio']), reverse=True)
    return results[:max_candidates]


def scan_delivery_candidates(
    tickers: Optional[List[str]] = None,
    start_date: str = '2020-01-01',
    end_date: Optional[Union[str, date, dt]] = None,
    max_candidates: int = 10,
) -> List[dict]:
    ticker_list = tickers or DEFAULT_NIFTY_200_TICKERS
    results = []

    for ticker in ticker_list:
        try:
            metrics = run_single_backtest(ticker, start_date=start_date, end_date=end_date)
        except Exception as exc:
            print(f'Skipping {ticker}: {exc}')
            continue

        if metrics and metrics['total_trades'] > 0:
            results.append(metrics)

    results.sort(
        key=lambda item: (
            item['accuracy_pct'],
            item['net_return_pct'],
            item['sharpe_ratio'] if item['sharpe_ratio'] is not None else float('-inf'),
        ),
        reverse=True,
    )
    return results[:max_candidates]


def scan_nifty200_candidates(
    tickers: Optional[List[str]] = None,
    start_date: str = '2020-01-01',
    end_date: Optional[Union[str, date, dt]] = None,
    max_candidates: int = 10,
) -> List[dict]:
    return scan_delivery_candidates(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        max_candidates=max_candidates,
    )


if __name__ == '__main__':
    print('Testing mother-candle strategy...')
    mother_candidates = scan_mother_candle_candidates(
        tickers=DEFAULT_NIFTY_100_TICKERS,
        start_date='2020-01-01',
        end_date=None,
        max_candidates=15,
    )
    total_tickers = len(DEFAULT_NIFTY_100_TICKERS)
    trades = sum(item['total_trades'] for item in mother_candidates)
    wins = sum(item['winning_trades'] for item in mother_candidates)
    accuracy = (wins / trades * 100.0) if trades else 0.0

    print(f'\nBacktested {total_tickers} stocks with the mother-candle swing strategy using data through today.')
    print(f'Generated {trades} trades across {len(mother_candidates)} stocks with an aggregate accuracy of {accuracy:.1f}%.')
    print('\nTop mother-candle candidates:')
    for item in mother_candidates:
        sharpe_value = item['sharpe_ratio']
        sharpe_text = f"{sharpe_value:.2f}" if sharpe_value is not None else 'n/a'
        print(
            f"{item['ticker']}: return={item['net_return_pct']:.2f}%, "
            f"accuracy={item['accuracy_pct']:.1f}%, sharpe={sharpe_text}, "
            f"trades={item['total_trades']}, wins={item['winning_trades']}, losses={item['losing_trades']}"
        )

    print('\nCurrent mother-candle watchlist (formed or forming):')
    current_candidates = scan_current_mother_candle_candidates(
        tickers=DEFAULT_NIFTY_100_TICKERS,
        start_date='2020-01-01',
        end_date=None,
        max_candidates=15,
    )
    if current_candidates:
        for item in current_candidates:
            print(
                f"{item['ticker']}: {item['status']} on {item['date']}, "
                f"high={item['high']:.2f}, low={item['low']:.2f}, "
                f"body_ratio={item['body_ratio']:.2f}, vol_ratio={item['volume_ratio']:.2f}"
            )
    else:
        print('No current mother-candle pattern detected in the sampled universe.')
