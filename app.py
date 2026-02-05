"""
Portfolio & Risk Management Dashboard
=====================================
A production-ready Streamlit dashboard for portfolio managers and valuation leads.
Features: Portfolio reconstruction, performance analytics, risk metrics (VaR/CVaR),
stress testing, rebalancing optimization, and Basel III-style reporting.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import io
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION & CUSTOM STYLING
# =============================================================================

st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and professional styling
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark theme overrides */
    .stApp {
        background-color: #0e1117;
    }

    /* Metric cards styling */
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    div[data-testid="metric-container"] > label {
        color: #8b8fa3;
        font-size: 0.85rem;
    }

    div[data-testid="metric-container"] > div {
        color: #ffffff;
        font-weight: 600;
    }

    /* Positive/negative delta colors */
    [data-testid="stMetricDelta"] svg {
        display: none;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e2130;
        border-radius: 10px;
        padding: 5px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #8b8fa3;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3d4263;
        color: #ffffff;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #151822;
        border-right: 1px solid #2d3250;
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #8b8fa3;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Data editor styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1e2130;
        border-radius: 8px;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }

    /* Info boxes */
    .stAlert {
        background-color: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 8px;
    }

    /* Custom KPI card class */
    .kpi-card {
        background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
        border: 1px solid #3d4263;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }

    .kpi-label {
        font-size: 0.9rem;
        color: #8b8fa3;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# AUTHENTICATION PLACEHOLDER
# =============================================================================
# To enable authentication, uncomment and configure streamlit-authenticator:
#
# import streamlit_authenticator as stauth
# from yaml.loader import SafeLoader
# import yaml
#
# with open('config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)
#
# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days']
# )
#
# name, authentication_status, username = authenticator.login('Login', 'main')
# if authentication_status:
#     authenticator.logout('Logout', 'sidebar')
#     st.sidebar.write(f'Welcome *{name}*')
#     # Main app code here
# elif authentication_status == False:
#     st.error('Username/password is incorrect')
# elif authentication_status == None:
#     st.warning('Please enter your username and password')

# =============================================================================
# DATA LOADING & SAMPLE DATA GENERATION
# =============================================================================

# Country Risk Premiums (Damodaran estimates - can be updated)
CRP_DATA = {
    'US': 0.0, 'USA': 0.0,
    'GB': 0.6, 'UK': 0.6,
    'DE': 0.0, 'GERMANY': 0.0,
    'FR': 0.5, 'FRANCE': 0.5,
    'JP': 0.8, 'JAPAN': 0.8,
    'CN': 1.0, 'CHINA': 1.0,
    'IN': 2.5, 'INDIA': 2.5,
    'BR': 3.5, 'BRAZIL': 3.5,
    'MX': 2.0, 'MEXICO': 2.0,
    'KR': 0.8, 'SOUTH KOREA': 0.8,
    'AU': 0.0, 'AUSTRALIA': 0.0,
    'CA': 0.0, 'CANADA': 0.0,
    'CH': 0.0, 'SWITZERLAND': 0.0,
    'HK': 0.6, 'HONG KONG': 0.6,
    'SG': 0.0, 'SINGAPORE': 0.0,
}

# Ticker to country mapping (common US stocks default to US)
TICKER_COUNTRY_MAP = {
    'AAPL': 'US', 'MSFT': 'US', 'GOOGL': 'US', 'AMZN': 'US', 'NVDA': 'US',
    'META': 'US', 'TSLA': 'US', 'JPM': 'US', 'V': 'US', 'JNJ': 'US',
    'SPY': 'US', 'QQQ': 'US', 'IWM': 'US', 'DIA': 'US', 'VTI': 'US',
    'EWJ': 'JP', 'FXI': 'CN', 'EWG': 'DE', 'EWU': 'GB', 'EEM': 'EM',
    'BABA': 'CN', 'TSM': 'TW', 'ASML': 'NL', 'NVO': 'DK', 'SAP': 'DE',
    # User's portfolio tickers
    'ADBE': 'US', 'CWEN': 'US', 'IVV': 'US', 'PG': 'US', 'UNH': 'US',
    'CEG': 'US', 'MCD': 'US', 'CASH': 'US', 'LULU': 'US',
}

# Portfolio configurations - add your portfolios here
# Note: initial_cash is now read from the holdings file (INITIAL_CASH row)
PORTFOLIO_CONFIG = {
    'EMF Portfolio': {
        'holdings_file': 'holdings.csv',
        'transactions_file': 'transactions.csv',
    },
    'DADCO Portfolio': {
        'holdings_file': 'Transactions and Holdings Files/HoldingsDADCO.csv',
        'transactions_file': 'Transactions and Holdings Files/TransactionsDADCO.csv',
    },
}


def load_portfolio_transactions(portfolio_name='Portfolio 1'):
    """Load transactions for the specified portfolio."""
    import os
    config = PORTFOLIO_CONFIG.get(portfolio_name, PORTFOLIO_CONFIG['EMF Portfolio'])
    default_path = os.path.join(os.path.dirname(__file__), config['transactions_file'])

    if os.path.exists(default_path):
        try:
            df = pd.read_csv(default_path)
            # Standardize column names
            column_mapping = {
                'Date': 'date', 'Ticker': 'symbol', 'Action': 'side',
                'Shares': 'quantity', 'Price': 'price', 'Fees': 'fees', 'Name': 'name'
            }
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            if 'side' in df.columns:
                df['side'] = df['side'].str.upper()
            return df.sort_values('date').reset_index(drop=True)
        except Exception as e:
            st.warning(f"Could not load {config['transactions_file']}: {e}")

    return generate_sample_transactions()


def load_portfolio_holdings(portfolio_name='Portfolio 1'):
    """Load holdings for the specified portfolio.

    Returns:
        tuple: (holdings_df, initial_cash) - Holdings DataFrame and initial cash from file
    """
    import os
    config = PORTFOLIO_CONFIG.get(portfolio_name, PORTFOLIO_CONFIG['EMF Portfolio'])

    # Try the configured file and fallback variations
    base_name = config['holdings_file'].replace('.csv', '')
    possible_paths = [
        os.path.join(os.path.dirname(__file__), config['holdings_file']),
        os.path.join(os.path.dirname(__file__), f'{base_name} (1).csv'),
        os.path.join(os.path.dirname(__file__), f'{base_name}-1.csv'),
    ]

    for default_path in possible_paths:
        if os.path.exists(default_path):
            try:
                df = pd.read_csv(default_path)
                # Standardize column names
                column_mapping = {
                    'Ticker': 'symbol', 'ticker': 'symbol',
                    'Shares': 'quantity', 'shares': 'quantity',
                }
                df = df.rename(columns=column_mapping)

                # Extract INITIAL_CASH from holdings if present
                initial_cash = 0.0
                if 'symbol' in df.columns:
                    initial_cash_row = df[df['symbol'] == 'INITIAL_CASH']
                    if not initial_cash_row.empty:
                        initial_cash = float(initial_cash_row['quantity'].iloc[0])
                        df = df[df['symbol'] != 'INITIAL_CASH']  # Remove from holdings

                if 'cost_basis' not in df.columns:
                    df['cost_basis'] = 0.0

                if 'country' not in df.columns:
                    df['country'] = df['symbol'].map(lambda x: TICKER_COUNTRY_MAP.get(x, 'US'))

                return df, initial_cash
            except Exception as e:
                st.warning(f"Could not load holdings file: {e}")

    return generate_sample_holdings(), 0.0


def load_default_transactions():
    """Load transactions from the default CSV file."""
    import os
    default_path = os.path.join(os.path.dirname(__file__), 'transactions.csv')

    if os.path.exists(default_path):
        try:
            df = pd.read_csv(default_path)
            # Standardize column names to match expected format
            column_mapping = {
                'Date': 'date',
                'Ticker': 'symbol',
                'Action': 'side',
                'Shares': 'quantity',
                'Price': 'price',
                'Fees': 'fees',
                'Name': 'name'
            }
            df = df.rename(columns=column_mapping)
            df['date'] = pd.to_datetime(df['date'])
            # Standardize side to uppercase
            df['side'] = df['side'].str.upper()
            return df.sort_values('date').reset_index(drop=True)
        except Exception as e:
            st.warning(f"Could not load transactions.csv: {e}")

    return generate_sample_transactions()


def load_default_holdings():
    """Load holdings from the default CSV file.

    Returns:
        tuple: (holdings_df, initial_cash) - Holdings DataFrame and initial cash from file
    """
    import os
    # Try different possible filenames
    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'holdings (1).csv'),
        os.path.join(os.path.dirname(__file__), 'holdings-1.csv'),
        os.path.join(os.path.dirname(__file__), 'holdings.csv'),
    ]

    for default_path in possible_paths:
        if os.path.exists(default_path):
            try:
                df = pd.read_csv(default_path)
                # Standardize column names
                column_mapping = {
                    'Ticker': 'symbol',
                    'Shares': 'quantity',
                    'ticker': 'symbol',
                    'shares': 'quantity',
                }
                df = df.rename(columns=column_mapping)

                # Extract INITIAL_CASH from holdings if present
                initial_cash = 0.0
                if 'symbol' in df.columns:
                    initial_cash_row = df[df['symbol'] == 'INITIAL_CASH']
                    if not initial_cash_row.empty:
                        initial_cash = float(initial_cash_row['quantity'].iloc[0])
                        df = df[df['symbol'] != 'INITIAL_CASH']  # Remove from holdings

                # Add cost_basis if not present (will be calculated from transactions or fetched)
                if 'cost_basis' not in df.columns:
                    df['cost_basis'] = 0.0  # Will be updated later

                # Add country if not present
                if 'country' not in df.columns:
                    df['country'] = df['symbol'].map(lambda x: TICKER_COUNTRY_MAP.get(x, 'US'))

                return df, initial_cash
            except Exception as e:
                st.warning(f"Could not load holdings file: {e}")

    return generate_sample_holdings(), 0.0


def calculate_cost_basis_from_transactions(transactions_df, holdings_df):
    """Calculate cost basis for each holding from transaction history."""
    if transactions_df is None or transactions_df.empty:
        return holdings_df

    holdings_df = holdings_df.copy()

    for idx, row in holdings_df.iterrows():
        symbol = row['symbol']
        # Get all transactions for this symbol
        symbol_txns = transactions_df[transactions_df['symbol'] == symbol].copy()

        if symbol_txns.empty:
            continue

        # Calculate weighted average cost from BUY transactions
        buy_txns = symbol_txns[symbol_txns['side'] == 'BUY']

        if not buy_txns.empty:
            total_cost = (buy_txns['quantity'] * buy_txns['price']).sum()
            total_shares = buy_txns['quantity'].sum()
            avg_cost = total_cost / total_shares if total_shares > 0 else 0
            holdings_df.at[idx, 'cost_basis'] = round(avg_cost, 2)
        else:
            # If no BUY transactions, use SELL price as estimate (position was held before)
            sell_txns = symbol_txns[symbol_txns['side'] == 'SELL']
            if not sell_txns.empty:
                # Use earliest sell price as cost basis estimate
                earliest_sell = sell_txns.sort_values('date').iloc[0]
                holdings_df.at[idx, 'cost_basis'] = round(earliest_sell['price'], 2)

    return holdings_df


def calculate_cash_from_transactions(transactions_df, initial_cash=0.0):
    """Calculate cash balance from transaction history.

    Supports transaction types:
        - BUY: Purchase securities (reduces cash)
        - SELL: Sell securities (increases cash)
        - WITHDRAWAL: Cash withdrawn from portfolio (reduces cash)
        - DEPOSIT: Cash added to portfolio (increases cash)
        - REBALANCE: Cash rebalancing/withdrawal (reduces cash, use negative amount for deposit)

    Args:
        transactions_df: DataFrame with columns: date, symbol, side, quantity, price, fees
                        For WITHDRAWAL/DEPOSIT/REBALANCE: use 'quantity' as the cash amount, price=1
        initial_cash: Starting cash balance

    Returns:
        tuple: (current_cash_balance, total_withdrawals, total_deposits)
    """
    if transactions_df is None or transactions_df.empty:
        return initial_cash, 0.0, 0.0

    cash_current = initial_cash
    total_withdrawals = 0.0
    total_deposits = 0.0

    # Sort transactions by date
    txns = transactions_df.sort_values('date').copy()

    for _, txn in txns.iterrows():
        quantity = float(txn.get('quantity', 0) or 0)
        price = float(txn.get('price', 0) or 0)
        fees = float(txn.get('fees', 0) or 0)
        side = str(txn.get('side', '')).upper()

        if side == 'SELL':
            # Selling adds cash (proceeds minus fees)
            cash_current += quantity * price - fees
        elif side == 'BUY':
            # Buying reduces cash (cost plus fees)
            cash_current -= quantity * price + fees
        elif side in ['WITHDRAWAL', 'REBALANCE']:
            # Withdrawal/rebalance removes cash from portfolio
            # quantity represents the cash amount being withdrawn
            withdrawal_amount = quantity * price if price > 0 else quantity
            cash_current -= withdrawal_amount
            total_withdrawals += withdrawal_amount
        elif side == 'DEPOSIT':
            # Deposit adds cash to portfolio
            deposit_amount = quantity * price if price > 0 else quantity
            cash_current += deposit_amount
            total_deposits += deposit_amount

    return round(cash_current, 2), round(total_withdrawals, 2), round(total_deposits, 2)


def estimate_cost_basis_from_prices(holdings_df, prices_df):
    """Estimate cost basis using the price at the beginning of the date range
    for positions without transaction history."""
    holdings_df = holdings_df.copy()

    for idx, row in holdings_df.iterrows():
        if row['cost_basis'] == 0 or pd.isna(row['cost_basis']):
            symbol = row['symbol']
            if symbol in prices_df.columns:
                historical_prices = prices_df[symbol].dropna()
                if len(historical_prices) > 0:
                    # Use the first price in the range as cost basis
                    holdings_df.at[idx, 'cost_basis'] = round(historical_prices.iloc[0], 2)

    return holdings_df


def generate_sample_transactions():
    """Generate sample transaction data for demonstration."""
    np.random.seed(42)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'JPM', 'V', 'JNJ']

    transactions = []
    start_date = datetime(2023, 1, 1)

    for symbol in symbols:
        # Initial buy
        buy_date = start_date + timedelta(days=np.random.randint(0, 30))
        qty = np.random.randint(10, 100)
        price = np.random.uniform(50, 500)
        transactions.append({
            'date': buy_date.strftime('%Y-%m-%d'),
            'symbol': symbol,
            'side': 'BUY',
            'quantity': qty,
            'price': round(price, 2),
            'fees': round(qty * price * 0.001, 2)
        })

        # Additional transactions
        for _ in range(np.random.randint(1, 4)):
            tx_date = buy_date + timedelta(days=np.random.randint(30, 365))
            side = np.random.choice(['BUY', 'SELL'], p=[0.6, 0.4])
            tx_qty = np.random.randint(5, 30)
            tx_price = price * (1 + np.random.uniform(-0.2, 0.3))
            transactions.append({
                'date': tx_date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'side': side,
                'quantity': tx_qty,
                'price': round(tx_price, 2),
                'fees': round(tx_qty * tx_price * 0.001, 2)
            })

    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def generate_sample_holdings():
    """Generate sample current holdings snapshot."""
    holdings = [
        {'symbol': 'AAPL', 'quantity': 75, 'cost_basis': 145.50, 'country': 'US'},
        {'symbol': 'MSFT', 'quantity': 50, 'cost_basis': 285.00, 'country': 'US'},
        {'symbol': 'GOOGL', 'quantity': 30, 'cost_basis': 125.00, 'country': 'US'},
        {'symbol': 'NVDA', 'quantity': 40, 'cost_basis': 420.00, 'country': 'US'},
        {'symbol': 'JPM', 'quantity': 60, 'cost_basis': 145.00, 'country': 'US'},
        {'symbol': 'V', 'quantity': 45, 'cost_basis': 235.00, 'country': 'US'},
        {'symbol': 'JNJ', 'quantity': 55, 'cost_basis': 158.00, 'country': 'US'},
    ]
    return pd.DataFrame(holdings)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_risk_free_rate():
    """Fetch the current 13-week T-bill rate (^IRX) as a proxy for the risk-free rate."""
    try:
        data = yf.download('^IRX', period='5d', progress=False)
        if not data.empty:
            # ^IRX returns the rate as a percentage (e.g. 4.2 means 4.2%)
            latest_rate = float(data['Close'].dropna().iloc[-1])
            return round(latest_rate, 2)
    except Exception:
        pass
    return 5.0  # Fallback default


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(symbols, start_date, end_date):
    """Fetch historical price data from Yahoo Finance."""
    try:
        if isinstance(symbols, str):
            symbols = [symbols]

        # Filter out empty symbols
        symbols = [s for s in symbols if s and isinstance(s, str) and s.strip()]

        if not symbols:
            return pd.DataFrame()

        data = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=True
        )

        if data.empty:
            return pd.DataFrame()

        # Handle single vs multiple symbols
        if len(symbols) == 1:
            if 'Close' in data.columns:
                result = pd.DataFrame({symbols[0]: data['Close']})
            else:
                result = pd.DataFrame({symbols[0]: data.iloc[:, 0]})
        else:
            if 'Close' in data.columns:
                result = data['Close']
            elif isinstance(data.columns, pd.MultiIndex):
                result = data.xs('Close', axis=1, level=0) if 'Close' in data.columns.get_level_values(0) else data
            else:
                result = data

        # Flatten MultiIndex columns if present
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = result.columns.get_level_values(-1)

        return result

    except Exception as e:
        st.error(f"Error fetching price data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fx_rates(currencies, start_date, end_date):
    """Fetch FX rates for currency conversion (vs USD)."""
    fx_tickers = {
        'EUR': 'EURUSD=X',
        'GBP': 'GBPUSD=X',
        'JPY': 'JPYUSD=X',
        'CNY': 'CNYUSD=X',
        'CHF': 'CHFUSD=X',
        'AUD': 'AUDUSD=X',
        'CAD': 'CADUSD=X',
    }

    rates = {}
    for curr in currencies:
        if curr in fx_tickers:
            try:
                data = yf.download(fx_tickers[curr], start=start_date, end=end_date, progress=False)
                rates[curr] = data['Close']
            except:
                rates[curr] = pd.Series(1.0, index=pd.date_range(start_date, end_date))

    return pd.DataFrame(rates)


# =============================================================================
# PORTFOLIO ANALYTICS FUNCTIONS
# =============================================================================

def reconstruct_portfolio_from_initial(transactions_df, initial_holdings_df, prices_df, initial_cash=0.0):
    """
    Reconstruct portfolio timeline from initial holdings + transactions.

    This properly tracks holdings through time by:
    1. Starting with initial holdings (from holdings.csv)
    2. Applying transactions in chronological order
    3. Tracking cash balance through time

    Args:
        transactions_df: DataFrame with columns: date, symbol, side, quantity, price, fees
        initial_holdings_df: DataFrame with initial holdings (symbol, quantity)
        prices_df: DataFrame with daily prices for each symbol
        initial_cash: Starting cash balance

    Returns:
        tuple: (portfolio_df, weights_df, cash_flows_df, current_holdings_df)
            - portfolio_df: Daily portfolio values with securities_value, cash, total_value
            - weights_df: Daily position weights
            - cash_flows_df: External cash flows (deposits/withdrawals) for TWR calculation
            - current_holdings_df: DataFrame with current holdings after all transactions
    """
    if prices_df is None or prices_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Initialize holdings from initial snapshot (excluding CASH row)
    holdings = {}
    for _, row in initial_holdings_df.iterrows():
        symbol = row['symbol']
        if symbol.upper() != 'CASH':
            holdings[symbol] = row['quantity']

    # Prepare transactions sorted by date
    if transactions_df is not None and not transactions_df.empty:
        txns = transactions_df.sort_values('date').copy()
        txns['date'] = pd.to_datetime(txns['date'])
    else:
        txns = pd.DataFrame(columns=['date', 'symbol', 'side', 'quantity', 'price', 'fees'])

    # Infer initial positions for symbols that appear in sell transactions
    # but aren't in initial holdings (to avoid phantom cash from selling non-existent positions)
    if not txns.empty:
        for symbol in txns['symbol'].unique():
            if not symbol or not isinstance(symbol, str) or symbol.upper() == 'CASH':
                continue
            symbol_txns = txns[txns['symbol'] == symbol].sort_values('date')
            running_qty = holdings.get(symbol, 0)
            min_qty = running_qty
            for _, txn in symbol_txns.iterrows():
                side = str(txn.get('side', '')).upper()
                qty = float(txn.get('quantity', 0) or 0)
                if side == 'BUY':
                    running_qty += qty
                elif side == 'SELL':
                    running_qty -= qty
                min_qty = min(min_qty, running_qty)
            if min_qty < 0:
                # Need additional shares in initial holdings to avoid going negative
                holdings[symbol] = holdings.get(symbol, 0) + abs(min_qty)

    # For symbols without price data (e.g. delisted), create synthetic prices
    # using the earliest transaction price so the position is properly valued
    # and sells convert position → cash without creating phantom value
    if not txns.empty:
        for symbol in txns['symbol'].unique():
            if not symbol or not isinstance(symbol, str) or symbol.upper() == 'CASH':
                continue
            if symbol not in prices_df.columns:
                # Use the earliest transaction price as a flat synthetic price
                symbol_txns = txns[txns['symbol'] == symbol].sort_values('date')
                first_price = float(symbol_txns.iloc[0]['price'])
                if first_price > 0:
                    prices_df[symbol] = first_price

    # Track cash flows for TWR calculation (external flows only: deposits/withdrawals)
    cash_flows = []  # List of (date, amount) - positive = inflow, negative = outflow

    # Build daily portfolio values
    portfolio_data = []
    cash_balance = initial_cash

    # Get all trading dates from prices
    dates = prices_df.index.tolist()
    first_price_date = pd.Timestamp(dates[0]).normalize() if dates else None

    # Create a dict to quickly lookup transactions by date
    txn_by_date = {}
    pre_period_txns = []  # Transactions BEFORE first price date

    for _, txn in txns.iterrows():
        txn_date = pd.Timestamp(txn['date']).normalize()
        if first_price_date and txn_date < first_price_date:
            # Transaction happened before our price data starts
            pre_period_txns.append(txn)
        else:
            if txn_date not in txn_by_date:
                txn_by_date[txn_date] = []
            txn_by_date[txn_date].append(txn)

    # Apply all transactions that happened BEFORE the first price date
    # These adjust our starting holdings but don't count as cash flows for TWR
    for txn in sorted(pre_period_txns, key=lambda x: x['date']):
        symbol = txn['symbol']
        side = txn['side'].upper()
        quantity = txn['quantity']
        price = txn['price']
        fees = txn.get('fees', 0) or 0

        if side == 'BUY':
            holdings[symbol] = holdings.get(symbol, 0) + quantity
            cash_balance -= (quantity * price + fees)
        elif side == 'SELL':
            holdings[symbol] = holdings.get(symbol, 0) - quantity
            cash_balance += (quantity * price - fees)
            if holdings.get(symbol, 0) <= 0:
                holdings.pop(symbol, None)
        elif side in ['WITHDRAWAL', 'REBALANCE']:
            amount = quantity * price if price > 0 else quantity
            cash_balance -= amount
            # Note: Pre-period withdrawals don't count as cash flows for TWR
        elif side == 'DEPOSIT':
            amount = quantity * price if price > 0 else quantity
            cash_balance += amount

    for date in dates:
        date_normalized = pd.Timestamp(date).normalize()

        # Apply any transactions on this date BEFORE calculating value
        if date_normalized in txn_by_date:
            for txn in txn_by_date[date_normalized]:
                symbol = txn['symbol']
                side = txn['side'].upper()
                quantity = txn['quantity']
                price = txn['price']
                fees = txn.get('fees', 0) or 0

                if side == 'BUY':
                    # Add to holdings, reduce cash
                    holdings[symbol] = holdings.get(symbol, 0) + quantity
                    cash_balance -= (quantity * price + fees)
                elif side == 'SELL':
                    # Reduce holdings, add to cash
                    holdings[symbol] = holdings.get(symbol, 0) - quantity
                    cash_balance += (quantity * price - fees)
                    # Remove symbol if zero holdings
                    if holdings.get(symbol, 0) <= 0:
                        holdings.pop(symbol, None)
                elif side in ['WITHDRAWAL', 'REBALANCE']:
                    # External cash outflow
                    amount = quantity * price if price > 0 else quantity
                    cash_balance -= amount
                    cash_flows.append({'date': date_normalized, 'amount': -amount})
                elif side == 'DEPOSIT':
                    # External cash inflow
                    amount = quantity * price if price > 0 else quantity
                    cash_balance += amount
                    cash_flows.append({'date': date_normalized, 'amount': amount})

        # Calculate securities value at end of day
        securities_value = 0
        position_values = {}

        for symbol, qty in holdings.items():
            if symbol in prices_df.columns and qty > 0:
                price = prices_df.loc[date, symbol]
                if pd.notna(price):
                    pos_value = qty * price
                    securities_value += pos_value
                    position_values[symbol] = pos_value

        total_value = securities_value + cash_balance  # Include cash (negative = margin/debt)

        portfolio_data.append({
            'date': date,
            'securities_value': securities_value,
            'cash': cash_balance,
            'portfolio_value': total_value,
            **{f'{s}_value': v for s, v in position_values.items()}
        })

    portfolio_df = pd.DataFrame(portfolio_data)
    portfolio_df.set_index('date', inplace=True)

    # Calculate weights
    weights_data = []
    for _, row in portfolio_df.iterrows():
        total = row['portfolio_value']
        if total > 0:
            weights = {}
            for col in portfolio_df.columns:
                if col.endswith('_value') and col != 'securities_value':
                    symbol = col.replace('_value', '')
                    weights[f'{symbol}_weight'] = row[col] / total if col in row else 0
            weights['cash_weight'] = row['cash'] / total if total > 0 else 0
            weights_data.append(weights)

    weights_df = pd.DataFrame(weights_data, index=portfolio_df.index)

    # Create cash flows DataFrame
    cash_flows_df = pd.DataFrame(cash_flows) if cash_flows else pd.DataFrame(columns=['date', 'amount'])

    # Create current holdings DataFrame (final state after all transactions)
    current_holdings_data = []
    for symbol, qty in holdings.items():
        if qty > 0:
            current_holdings_data.append({
                'symbol': symbol,
                'quantity': qty,
                'cost_basis': 0.0,  # Will be calculated separately
                'country': TICKER_COUNTRY_MAP.get(symbol, 'US')
            })
    # Add cash as a position
    if cash_balance > 0:
        current_holdings_data.append({
            'symbol': 'CASH',
            'quantity': cash_balance,
            'cost_basis': 1.0,
            'country': 'US'
        })
    current_holdings_df = pd.DataFrame(current_holdings_data)

    return portfolio_df, weights_df, cash_flows_df, current_holdings_df


def calculate_time_weighted_return(portfolio_df, cash_flows_df):
    """
    Calculate Time-Weighted Return (TWR).

    TWR eliminates the impact of cash flows on performance measurement by:
    1. Splitting the period into sub-periods at each cash flow
    2. Calculating the return for each sub-period
    3. Geometrically linking all sub-period returns

    Formula: TWR = [(1 + R1) × (1 + R2) × ... × (1 + Rn)] - 1

    Args:
        portfolio_df: DataFrame with 'portfolio_value' column and date index
        cash_flows_df: DataFrame with 'date' and 'amount' columns for external flows

    Returns:
        dict: {
            'twr_total': Total time-weighted return,
            'twr_annualized': Annualized TWR,
            'sub_period_returns': List of sub-period returns,
            'daily_twr': Series of daily TWR values for charting
        }
    """
    if portfolio_df.empty or 'portfolio_value' not in portfolio_df.columns:
        return {'twr_total': 0, 'twr_annualized': 0, 'sub_period_returns': [], 'daily_twr': pd.Series()}

    values = portfolio_df['portfolio_value'].dropna()
    if len(values) < 2:
        return {'twr_total': 0, 'twr_annualized': 0, 'sub_period_returns': [], 'daily_twr': pd.Series()}

    # Get cash flow dates
    if cash_flows_df is not None and not cash_flows_df.empty:
        flow_dates = set(pd.to_datetime(cash_flows_df['date']).dt.normalize())
        flow_amounts = cash_flows_df.set_index(pd.to_datetime(cash_flows_df['date']).dt.normalize())['amount'].to_dict()
    else:
        flow_dates = set()
        flow_amounts = {}

    # Calculate daily returns, adjusting for cash flows
    daily_returns = []
    dates = values.index.tolist()

    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]

        prev_value = values.iloc[i-1]
        curr_value = values.iloc[i]

        # Check if there was a cash flow on the current date
        curr_date_normalized = pd.Timestamp(curr_date).normalize()

        if curr_date_normalized in flow_dates:
            # Cash flow happened at START of this day (before market movement)
            # In reconstruct_portfolio_from_initial, transactions are processed BEFORE
            # calculating the day's ending value.
            #
            # Correct TWR formula for start-of-day cash flow:
            # R = V_end / (V_start + CF) - 1
            #
            # Where:
            # - V_start = previous day's ending value
            # - CF = cash flow (positive for deposit, negative for withdrawal)
            # - V_end = current day's ending value (already reflects the CF in cash balance)
            #
            # This measures return on the capital that was actually invested during the day
            cf = flow_amounts.get(curr_date_normalized, 0)
            adjusted_start_value = prev_value + cf  # Capital at start of day after CF
            if adjusted_start_value > 0:
                ret = (curr_value / adjusted_start_value) - 1
            else:
                ret = 0
        else:
            # No cash flow - simple return
            if prev_value > 0:
                ret = (curr_value / prev_value) - 1
            else:
                ret = 0

        daily_returns.append({'date': curr_date, 'return': ret})

    returns_df = pd.DataFrame(daily_returns).set_index('date')

    # Calculate cumulative TWR (geometric linking)
    cumulative_twr = (1 + returns_df['return']).cumprod()

    # Total TWR
    twr_total = cumulative_twr.iloc[-1] - 1 if len(cumulative_twr) > 0 else 0

    # Annualized TWR
    n_days = (dates[-1] - dates[0]).days
    n_years = n_days / 365.25 if n_days > 0 else 1
    twr_annualized = (1 + twr_total) ** (1 / n_years) - 1 if n_years > 0 else twr_total

    return {
        'twr_total': twr_total,
        'twr_annualized': twr_annualized,
        'sub_period_returns': returns_df['return'].tolist(),
        'daily_twr': cumulative_twr - 1  # Convert to return series
    }


def calculate_money_weighted_return(portfolio_df, cash_flows_df, initial_cash=0.0):
    """
    Calculate Money-Weighted Return (MWR) using Internal Rate of Return (IRR).

    MWR accounts for the timing and size of cash flows, reflecting the
    investor's actual experience. It solves for the discount rate r where:

    NPV = CF_0 + CF_1/(1+r)^t1 + CF_2/(1+r)^t2 + ... + V_final/(1+r)^T = 0

    Args:
        portfolio_df: DataFrame with 'portfolio_value' column and date index
        cash_flows_df: DataFrame with 'date' and 'amount' columns
        initial_cash: Initial investment amount

    Returns:
        dict: {
            'mwr_total': Total money-weighted return,
            'mwr_annualized': Annualized MWR (IRR),
            'xirr': Same as mwr_annualized (industry term)
        }
    """
    if portfolio_df.empty or 'portfolio_value' not in portfolio_df.columns:
        return {'mwr_total': 0, 'mwr_annualized': 0, 'xirr': 0}

    values = portfolio_df['portfolio_value'].dropna()
    if len(values) < 2:
        return {'mwr_total': 0, 'mwr_annualized': 0, 'xirr': 0}

    # Build cash flow series for XIRR calculation
    # Convention: negative = outflow (investment), positive = inflow (withdrawal/final value)
    cash_flow_list = []

    # Initial investment (outflow - negative)
    start_date = values.index[0]
    initial_value = values.iloc[0]
    # The initial investment is the starting portfolio value
    cash_flow_list.append({'date': start_date, 'amount': -initial_value})

    # Intermediate cash flows (deposits are outflows/negative, withdrawals are inflows/positive)
    if cash_flows_df is not None and not cash_flows_df.empty:
        for _, row in cash_flows_df.iterrows():
            # In cash_flows_df: positive = deposit (into portfolio), negative = withdrawal
            # For IRR: we flip the sign (deposit = investor outflow = negative)
            cash_flow_list.append({
                'date': pd.Timestamp(row['date']),
                'amount': -row['amount']  # Flip sign for IRR convention
            })

    # Final value (inflow - positive)
    end_date = values.index[-1]
    final_value = values.iloc[-1]
    cash_flow_list.append({'date': end_date, 'amount': final_value})

    # Sort by date
    cash_flow_list.sort(key=lambda x: x['date'])

    # Calculate XIRR using Newton-Raphson method
    def xnpv(rate, cash_flows):
        """Calculate NPV with exact dates."""
        if rate <= -1:
            return float('inf')
        t0 = cash_flows[0]['date']
        total = 0
        for cf in cash_flows:
            days = (cf['date'] - t0).days
            years = days / 365.25
            total += cf['amount'] / ((1 + rate) ** years)
        return total

    def xirr(cash_flows, guess=0.1):
        """Calculate IRR using Newton-Raphson."""
        rate = guess
        for _ in range(100):  # Max iterations
            npv = xnpv(rate, cash_flows)

            # Numerical derivative
            delta = 0.0001
            npv_delta = xnpv(rate + delta, cash_flows)
            derivative = (npv_delta - npv) / delta

            if abs(derivative) < 1e-10:
                break

            new_rate = rate - npv / derivative

            # Bound the rate to reasonable values
            new_rate = max(-0.99, min(10, new_rate))

            if abs(new_rate - rate) < 1e-7:
                return new_rate
            rate = new_rate

        return rate

    try:
        mwr_annualized = xirr(cash_flow_list)
    except:
        mwr_annualized = 0

    # Calculate total MWR over the period
    n_days = (values.index[-1] - values.index[0]).days
    n_years = n_days / 365.25 if n_days > 0 else 1
    mwr_total = (1 + mwr_annualized) ** n_years - 1

    return {
        'mwr_total': mwr_total,
        'mwr_annualized': mwr_annualized,
        'xirr': mwr_annualized
    }


def calculate_adjusted_benchmark(benchmark_prices, portfolio_df, cash_flows_df):
    """
    Calculate benchmark returns adjusted for the same cash flows as the portfolio.

    When the portfolio has withdrawals/deposits, we adjust the benchmark by
    simulating the same cash flows to enable fair comparison.

    Args:
        benchmark_prices: Series of benchmark prices
        portfolio_df: Portfolio DataFrame with 'portfolio_value'
        cash_flows_df: DataFrame with 'date' and 'amount' columns

    Returns:
        dict: {
            'adjusted_benchmark_values': Series of adjusted benchmark values,
            'adjusted_benchmark_returns': Series of returns,
            'twr': Time-weighted return of adjusted benchmark,
            'raw_benchmark_return': Unadjusted benchmark return for comparison
        }
    """
    if benchmark_prices.empty or portfolio_df.empty:
        return {
            'adjusted_benchmark_values': pd.Series(),
            'adjusted_benchmark_returns': pd.Series(),
            'twr': 0,
            'raw_benchmark_return': 0
        }

    # Align dates
    common_dates = portfolio_df.index.intersection(benchmark_prices.index)
    if len(common_dates) == 0:
        return {
            'adjusted_benchmark_values': pd.Series(),
            'adjusted_benchmark_returns': pd.Series(),
            'twr': 0,
            'raw_benchmark_return': 0
        }

    benchmark_aligned = benchmark_prices.loc[common_dates]
    portfolio_aligned = portfolio_df.loc[common_dates]

    # Get initial portfolio value to set benchmark starting value
    initial_portfolio_value = portfolio_aligned['portfolio_value'].iloc[0]

    # Calculate benchmark "units" (like shares of benchmark ETF)
    initial_benchmark_price = benchmark_aligned.iloc[0]
    benchmark_units = initial_portfolio_value / initial_benchmark_price

    # Build cash flow lookup
    if cash_flows_df is not None and not cash_flows_df.empty:
        flow_lookup = {}
        for _, row in cash_flows_df.iterrows():
            date = pd.Timestamp(row['date']).normalize()
            flow_lookup[date] = flow_lookup.get(date, 0) + row['amount']
    else:
        flow_lookup = {}

    # Calculate adjusted benchmark values
    adjusted_values = []

    for date in common_dates:
        date_normalized = pd.Timestamp(date).normalize()
        benchmark_price = benchmark_aligned.loc[date]

        # Check for cash flow on this date
        if date_normalized in flow_lookup:
            cf = flow_lookup[date_normalized]
            # Positive cf = deposit (add units), negative = withdrawal (remove units)
            units_change = cf / benchmark_price
            benchmark_units += units_change

        adjusted_value = benchmark_units * benchmark_price
        adjusted_values.append({'date': date, 'value': adjusted_value})

    adjusted_df = pd.DataFrame(adjusted_values).set_index('date')
    adjusted_benchmark_values = adjusted_df['value']

    # Calculate returns
    adjusted_benchmark_returns = adjusted_benchmark_values.pct_change().dropna()

    # Calculate raw benchmark return (price return only, no cash flows)
    raw_return = (benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0]) - 1

    # Calculate time period
    n_days = (common_dates[-1] - common_dates[0]).days
    n_years = n_days / 365.25 if n_days > 0 else 1

    # Raw annualized return (benchmark with no cash flows)
    raw_annualized = (1 + raw_return) ** (1 / n_years) - 1 if n_years > 0 else raw_return

    # Calculate cash-flow adjusted benchmark TWR
    # This is what return you'd get if you invested in benchmark with same cash flows
    # We need to calculate TWR for the adjusted benchmark values
    if len(adjusted_benchmark_values) > 1:
        # Calculate daily returns for adjusted benchmark
        adj_daily_returns = adjusted_benchmark_values.pct_change().dropna()

        # Adjust returns on cash flow days (same logic as portfolio TWR)
        for date_norm, cf in flow_lookup.items():
            if date_norm in adj_daily_returns.index:
                # On cash flow days, the return should exclude the cash flow effect
                # This is already handled since we track units, not just values
                pass

        # Calculate TWR from adjusted returns
        adj_total_return = (1 + adj_daily_returns).prod() - 1
        adj_annualized = (1 + adj_total_return) ** (1 / n_years) - 1 if n_years > 0 else adj_total_return
    else:
        adj_total_return = raw_return
        adj_annualized = raw_annualized

    return {
        'adjusted_benchmark_values': adjusted_benchmark_values,
        'adjusted_benchmark_returns': adjusted_benchmark_returns,
        'adjusted_total': adj_total_return,  # Total cash-flow adjusted benchmark return
        'adjusted_annualized': adj_annualized,  # Annualized cash-flow adjusted benchmark return
        'twr': raw_annualized,  # Raw benchmark TWR (for reference)
        'raw_total': raw_return,  # Total raw benchmark return
        'raw_benchmark_return': raw_return,
        'raw_annualized': raw_annualized
    }


def reconstruct_portfolio(transactions_df, holdings_df, prices_df):
    """
    Legacy function: Reconstruct portfolio using current holdings snapshot.

    NOTE: This function uses the FINAL holdings state and applies it across all dates.
    For proper portfolio reconstruction that tracks holdings through time based on
    transactions, use reconstruct_portfolio_from_initial() instead.

    This function is kept for backward compatibility with position metrics calculation.
    """
    if transactions_df is None or transactions_df.empty:
        transactions_df = pd.DataFrame(columns=['date', 'symbol', 'side', 'quantity', 'price', 'fees'])

    symbols = holdings_df['symbol'].unique().tolist()

    # Initialize holdings from current snapshot
    current_holdings = holdings_df.set_index('symbol')['quantity'].to_dict()
    cost_basis = holdings_df.set_index('symbol')['cost_basis'].to_dict()

    # Get price data
    if prices_df is None or prices_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Calculate daily portfolio value
    portfolio_values = []

    for date in prices_df.index:
        daily_value = 0
        for symbol in symbols:
            if symbol in prices_df.columns and symbol in current_holdings:
                price = prices_df.loc[date, symbol]
                if pd.notna(price):
                    daily_value += current_holdings[symbol] * price

        portfolio_values.append({
            'date': date,
            'portfolio_value': daily_value
        })

    portfolio_df = pd.DataFrame(portfolio_values)
    portfolio_df.set_index('date', inplace=True)

    # Calculate weights
    weights_data = []
    for date in prices_df.index:
        total_value = 0
        position_values = {}

        for symbol in symbols:
            if symbol in prices_df.columns and symbol in current_holdings:
                price = prices_df.loc[date, symbol]
                if pd.notna(price):
                    pos_value = current_holdings[symbol] * price
                    position_values[symbol] = pos_value
                    total_value += pos_value

        if total_value > 0:
            weights = {f'{s}_weight': v / total_value for s, v in position_values.items()}
            weights['date'] = date
            weights_data.append(weights)

    weights_df = pd.DataFrame(weights_data)
    if not weights_df.empty:
        weights_df.set_index('date', inplace=True)

    return portfolio_df, weights_df


def calculate_returns(portfolio_df):
    """Calculate portfolio returns time series."""
    if portfolio_df.empty:
        return pd.Series(dtype=float)

    returns = portfolio_df['portfolio_value'].pct_change().dropna()
    return returns


def calculate_performance_metrics(returns, benchmark_returns=None, risk_free_rate=0.05):
    """Calculate comprehensive performance metrics."""
    if returns.empty:
        return {}

    # Align returns
    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner')
        aligned.columns = ['portfolio', 'benchmark']
        returns = aligned['portfolio']
        benchmark_returns = aligned['benchmark']

    # Basic metrics
    total_return = (1 + returns).prod() - 1
    trading_days = 252
    n_years = len(returns) / trading_days
    annualized_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # Risk metrics
    volatility = returns.std() * np.sqrt(trading_days)

    # Sharpe Ratio
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else volatility
    sortino_ratio = excess_return / downside_std if downside_std > 0 else 0

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'positive_days': (returns > 0).mean(),
        'best_day': returns.max(),
        'worst_day': returns.min(),
        'avg_daily_return': returns.mean(),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
    }

    # Benchmark-relative metrics
    if benchmark_returns is not None and not benchmark_returns.empty:
        # Beta
        covariance = np.cov(returns.values, benchmark_returns.values)[0, 1]
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1

        # Alpha (Jensen's)
        benchmark_ann_return = (1 + benchmark_returns).prod() ** (trading_days / len(benchmark_returns)) - 1
        alpha = annualized_return - (risk_free_rate + beta * (benchmark_ann_return - risk_free_rate))

        # R-squared
        correlation = returns.corr(benchmark_returns)
        r_squared = correlation ** 2

        # Tracking Error
        tracking_diff = returns - benchmark_returns
        tracking_error = tracking_diff.std() * np.sqrt(trading_days)

        # Information Ratio
        information_ratio = (annualized_return - benchmark_ann_return) / tracking_error if tracking_error > 0 else 0

        metrics.update({
            'beta': beta,
            'alpha': alpha,
            'r_squared': r_squared,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'benchmark_return': benchmark_ann_return,
        })

    return metrics


def calculate_var_cvar(returns, confidence=0.95, method='historical'):
    """Calculate Value at Risk and Conditional VaR."""
    if returns.empty:
        return 0, 0

    if method == 'historical':
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var].mean()
    elif method == 'parametric':
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - confidence)
        var = mean + z_score * std
        cvar = mean - std * stats.norm.pdf(z_score) / (1 - confidence)

    return var, cvar


def monte_carlo_simulation(returns, n_simulations=10000, n_days=252, initial_value=100000):
    """Run Monte Carlo simulation for portfolio."""
    if returns.empty:
        return np.array([]), np.array([])

    mean_return = returns.mean()
    std_return = returns.std()

    simulations = np.zeros((n_simulations, n_days))
    simulations[:, 0] = initial_value

    for t in range(1, n_days):
        random_returns = np.random.normal(mean_return, std_return, n_simulations)
        simulations[:, t] = simulations[:, t-1] * (1 + random_returns)

    final_values = simulations[:, -1]

    return simulations, final_values


def calculate_correlation_matrix(prices_df):
    """Calculate correlation matrix from price data."""
    if prices_df.empty:
        return pd.DataFrame()

    returns = prices_df.pct_change().dropna()
    return returns.corr()


def calculate_position_metrics(holdings_df, prices_df, benchmark_returns=None):
    """Calculate metrics for each position."""
    if holdings_df.empty or prices_df.empty:
        return pd.DataFrame()

    position_data = []

    for _, row in holdings_df.iterrows():
        symbol = row['symbol']
        quantity = row['quantity']
        cost_basis = row.get('cost_basis', 0)

        # Special handling for CASH
        if symbol.upper() == 'CASH':
            market_value = quantity  # Cash quantity IS the value
            position_data.append({
                'Symbol': 'CASH',
                'Quantity': quantity,
                'Cost Basis': 1.0,
                'Current Price': 1.0,
                'Market Value': market_value,
                'Unrealized P&L': 0,
                'P&L %': 0,
                'Weight': 0,
                'Volatility': 0,
                'Beta': 0,
                'Country': 'US',
                'CRP (%)': 0,
            })
            continue

        if symbol in prices_df.columns:
            price_series = prices_df[symbol].dropna()
            if price_series.empty:
                continue

            current_price = price_series.iloc[-1]

            # Handle NaN or zero current price
            if pd.isna(current_price) or current_price == 0:
                continue

            returns = price_series.pct_change().dropna()

            # Calculate position metrics
            market_value = quantity * current_price

            # Handle zero or missing cost basis
            if cost_basis is None or cost_basis == 0 or pd.isna(cost_basis):
                # Use current price as cost basis estimate
                cost_basis = current_price
                total_cost = market_value
                unrealized_pnl = 0
                pnl_pct = 0
            else:
                total_cost = quantity * cost_basis
                unrealized_pnl = market_value - total_cost
                pnl_pct = (current_price / cost_basis - 1) * 100 if cost_basis > 0 else 0

            # Volatility
            vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

            # Beta vs benchmark
            beta = 1.0
            if benchmark_returns is not None and not benchmark_returns.empty and len(returns) > 0:
                aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner')
                if len(aligned) > 10:
                    try:
                        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
                        var = aligned.iloc[:, 1].var()
                        beta = cov / var if var > 0 else 1.0
                    except:
                        beta = 1.0

            # Country and CRP
            country = row.get('country', TICKER_COUNTRY_MAP.get(symbol, 'US'))
            if pd.isna(country):
                country = 'US'
            crp = CRP_DATA.get(str(country).upper(), 0)

            position_data.append({
                'Symbol': symbol,
                'Quantity': quantity,
                'Cost Basis': cost_basis,
                'Current Price': current_price,
                'Market Value': market_value,
                'Unrealized P&L': unrealized_pnl,
                'P&L %': pnl_pct,
                'Weight': 0,  # Will be calculated after
                'Volatility': vol,
                'Beta': beta,
                'Country': country,
                'CRP (%)': crp,
            })

    df = pd.DataFrame(position_data)

    # Calculate weights
    if not df.empty:
        total_value = df['Market Value'].sum()
        if total_value > 0:
            df['Weight'] = df['Market Value'] / total_value * 100
        else:
            df['Weight'] = 0

    return df


# =============================================================================
# STRESS TESTING FUNCTIONS
# =============================================================================

def run_stress_tests(holdings_df, prices_df, scenarios=None):
    """Run stress test scenarios on portfolio."""
    if scenarios is None:
        scenarios = {
            'Market Crash (-30%)': {'market': -0.30},
            'Tech Selloff (-25% tech, -10% others)': {'tech': -0.25, 'other': -0.10},
            'Interest Rate Shock': {'rate_sensitive': -0.15, 'growth': -0.20},
            'Currency Crisis (EM -40%)': {'em': -0.40, 'dm': -0.05},
            'Inflation Surge': {'growth': -0.15, 'value': 0.05},
            'Black Swan (-50%)': {'market': -0.50},
        }

    tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'ADBE']
    em_symbols = ['BABA', 'FXI', 'EEM']
    cash_symbols = ['CASH']  # Cash is not affected by market shocks

    results = []

    for scenario_name, shocks in scenarios.items():
        total_loss = 0

        for _, row in holdings_df.iterrows():
            symbol = row['symbol']
            quantity = row['quantity']

            # CASH is not affected by market shocks
            if symbol.upper() == 'CASH':
                continue

            if symbol in prices_df.columns:
                current_price = prices_df[symbol].iloc[-1]
                position_value = quantity * current_price

                # Apply appropriate shock
                shock = shocks.get('market', 0)
                if symbol in tech_symbols and 'tech' in shocks:
                    shock = shocks['tech']
                elif symbol in em_symbols and 'em' in shocks:
                    shock = shocks['em']
                elif 'other' in shocks and symbol not in tech_symbols:
                    shock = shocks['other']
                elif 'dm' in shocks and symbol not in em_symbols:
                    shock = shocks['dm']

                total_loss += position_value * shock

        # Calculate total portfolio including cash
        total_portfolio = 0
        for _, row in holdings_df.iterrows():
            symbol = row['symbol']
            if symbol.upper() == 'CASH':
                total_portfolio += row['quantity']
            elif symbol in prices_df.columns:
                total_portfolio += row['quantity'] * prices_df[symbol].iloc[-1]

        results.append({
            'Scenario': scenario_name,
            'Portfolio Loss ($)': total_loss,
            'Portfolio Loss (%)': (total_loss / total_portfolio * 100) if total_portfolio > 0 else 0,
        })

    return pd.DataFrame(results)


# =============================================================================
# REBALANCING OPTIMIZER
# =============================================================================

def optimize_portfolio(returns_df, method='max_sharpe', risk_free_rate=0.05,
                       target_return=None, max_weight=0.25, min_weight=0.02):
    """Optimize portfolio weights using mean-variance optimization."""
    if returns_df.empty:
        return {}

    # Calculate expected returns and covariance
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    n_assets = len(mean_returns)

    # Objective functions
    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights

    def portfolio_return(weights):
        return weights.T @ mean_returns

    def neg_sharpe_ratio(weights):
        ret = portfolio_return(weights)
        vol = np.sqrt(portfolio_variance(weights))
        return -(ret - risk_free_rate) / vol if vol > 0 else 0

    def min_variance(weights):
        return portfolio_variance(weights)

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
    ]

    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: portfolio_return(w) - target_return
        })

    # Bounds
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

    # Initial guess (equal weight)
    init_weights = np.array([1/n_assets] * n_assets)

    # Select objective
    if method == 'max_sharpe':
        objective = neg_sharpe_ratio
    elif method == 'min_variance':
        objective = min_variance
    elif method == 'risk_parity':
        def risk_parity_obj(weights):
            port_vol = np.sqrt(portfolio_variance(weights))
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / port_vol
            target_risk = port_vol / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)
        objective = risk_parity_obj
    else:
        objective = neg_sharpe_ratio

    # Optimize
    try:
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if result.success:
            optimal_weights = dict(zip(returns_df.columns, result.x))
            opt_return = portfolio_return(result.x)
            opt_vol = np.sqrt(portfolio_variance(result.x))
            opt_sharpe = (opt_return - risk_free_rate) / opt_vol if opt_vol > 0 else 0

            return {
                'weights': optimal_weights,
                'expected_return': opt_return,
                'expected_volatility': opt_vol,
                'sharpe_ratio': opt_sharpe,
                'success': True
            }
    except Exception as e:
        st.error(f"Optimization failed: {e}")

    return {'success': False}


def generate_rebalance_transactions(current_holdings, optimal_weights, prices_df, total_value):
    """Generate transactions needed to rebalance to optimal weights."""
    transactions = []

    for symbol, target_weight in optimal_weights.items():
        target_value = total_value * target_weight
        current_qty = current_holdings.get(symbol, 0)

        if symbol in prices_df.columns:
            current_price = prices_df[symbol].iloc[-1]
            current_value = current_qty * current_price
            value_diff = target_value - current_value
            qty_diff = int(value_diff / current_price)

            if abs(qty_diff) > 0:
                transactions.append({
                    'Symbol': symbol,
                    'Side': 'BUY' if qty_diff > 0 else 'SELL',
                    'Quantity': abs(qty_diff),
                    'Est. Price': current_price,
                    'Est. Value': abs(qty_diff) * current_price,
                    'Current Weight': (current_value / total_value * 100) if total_value > 0 else 0,
                    'Target Weight': target_weight * 100,
                })

    return pd.DataFrame(transactions)


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_to_excel(data_dict, filename='portfolio_report.xlsx'):
    """Export multiple DataFrames to Excel."""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in data_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name[:31], index=True)
            elif isinstance(df, dict):
                pd.DataFrame([df]).to_excel(writer, sheet_name=sheet_name[:31], index=False)

    return output.getvalue()


def generate_pdf_report(metrics, holdings_df, stress_results):
    """Generate a simple PDF report (requires reportlab)."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        elements.append(Paragraph("Portfolio Risk Report", styles['Title']))
        elements.append(Spacer(1, 20))

        # Date
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 20))

        # Key Metrics
        elements.append(Paragraph("Key Performance Metrics", styles['Heading2']))
        metrics_data = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in metrics.items()]
        metrics_table = Table(metrics_data, colWidths=[200, 150])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 20))

        doc.build(elements)
        return output.getvalue()
    except ImportError:
        return None


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Initialize session state - Load from CSV files if available
    if 'selected_portfolio' not in st.session_state:
        st.session_state.selected_portfolio = list(PORTFOLIO_CONFIG.keys())[0]

    selected_portfolio = st.session_state.selected_portfolio

    if 'transactions' not in st.session_state:
        st.session_state.transactions = load_portfolio_transactions(selected_portfolio)
    if 'initial_cash' not in st.session_state:
        st.session_state.initial_cash = 0.0
    if 'holdings' not in st.session_state:
        st.session_state.holdings, st.session_state.initial_cash = load_portfolio_holdings(selected_portfolio)
        # Calculate cost basis from transactions if not provided
        if st.session_state.holdings['cost_basis'].sum() == 0:
            st.session_state.holdings = calculate_cost_basis_from_transactions(
                st.session_state.transactions,
                st.session_state.holdings
            )
        # Calculate cash from transactions
        initial_cash = st.session_state.initial_cash
        if initial_cash > 0 or st.session_state.transactions is not None:
            calculated_cash, total_withdrawals, total_deposits = calculate_cash_from_transactions(
                st.session_state.transactions,
                initial_cash
            )
            holdings_df = st.session_state.holdings.copy()
            if 'CASH' in holdings_df['symbol'].values:
                holdings_df.loc[holdings_df['symbol'] == 'CASH', 'quantity'] = calculated_cash
            elif calculated_cash != 0:
                new_cash_row = pd.DataFrame([{
                    'symbol': 'CASH',
                    'quantity': calculated_cash,
                    'cost_basis': 1.0,
                    'country': 'US'
                }])
                holdings_df = pd.concat([holdings_df, new_cash_row], ignore_index=True)
            st.session_state.holdings = holdings_df
    if 'prices_loaded' not in st.session_state:
        st.session_state.prices_loaded = False

    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="margin: 0; font-size: 2.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
             UO MIG Portfolio Risk Dashboard
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # SIDEBAR
    # ==========================================================================
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # Portfolio Selector
        st.markdown("### 📂 Portfolio Selection")
        portfolio_options = list(PORTFOLIO_CONFIG.keys())

        # Initialize selected portfolio in session state
        if 'selected_portfolio' not in st.session_state:
            st.session_state.selected_portfolio = portfolio_options[0]

        selected_portfolio = st.selectbox(
            "Select Portfolio",
            options=portfolio_options,
            index=portfolio_options.index(st.session_state.selected_portfolio),
            help="Switch between different portfolios"
        )

        # Reload data when portfolio changes
        if selected_portfolio != st.session_state.selected_portfolio:
            st.session_state.selected_portfolio = selected_portfolio
            st.session_state.transactions = load_portfolio_transactions(selected_portfolio)
            st.session_state.holdings, st.session_state.initial_cash = load_portfolio_holdings(selected_portfolio)
            # Calculate cost basis from transactions
            if st.session_state.holdings['cost_basis'].sum() == 0:
                st.session_state.holdings = calculate_cost_basis_from_transactions(
                    st.session_state.transactions,
                    st.session_state.holdings
                )
            # Calculate cash from transactions (initial_cash now from holdings file)
            initial_cash = st.session_state.initial_cash
            if initial_cash > 0 or st.session_state.transactions is not None:
                calculated_cash, _, _ = calculate_cash_from_transactions(
                    st.session_state.transactions,
                    initial_cash
                )
                holdings_df = st.session_state.holdings.copy()
                if 'CASH' in holdings_df['symbol'].values:
                    holdings_df.loc[holdings_df['symbol'] == 'CASH', 'quantity'] = calculated_cash
                elif calculated_cash != 0:
                    new_cash_row = pd.DataFrame([{
                        'symbol': 'CASH',
                        'quantity': calculated_cash,
                        'cost_basis': 1.0,
                        'country': 'US'
                    }])
                    holdings_df = pd.concat([holdings_df, new_cash_row], ignore_index=True)
                st.session_state.holdings = holdings_df
            st.session_state.prices_loaded = False
            st.rerun()

        st.markdown("---")

        # Date range - default to cover transaction history
        st.markdown("### 📅 Date Range")

        # Fixed start date as per requirement: March 31, 2025
        default_start = datetime(2025, 3, 31)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                value=default_start,
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End",
                value=datetime.now(),
                max_value=datetime.now()
            )

        st.markdown("---")

        # Benchmark
        st.markdown("### 📈 Benchmark")
        benchmark = st.selectbox(
            "Select Benchmark",
            options=['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'AGG', 'IEMG'],
            index=0,
            help="Benchmark for relative performance metrics"
        )

        st.markdown("---")

        # Risk parameters
        st.markdown("### ⚠️ Risk Parameters")
        var_confidence = st.slider(
            "VaR Confidence Level",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01,
            format="%.2f"
        )

        enable_crp = st.toggle(
            "Enable Country Risk Premium",
            value=True,
            help="Adjust returns for country-specific risk premiums"
        )

        position_limit = st.slider(
            "Max Position Weight (%)",
            min_value=5,
            max_value=50,
            value=25,
            step=5
        )

        st.markdown("---")

        # Risk-free rate (auto-fetched from 13-week T-bill)
        live_rf_rate = fetch_risk_free_rate()
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=live_rf_rate,
            step=0.1,
            help=f"Auto-fetched from 13-week T-bill (^IRX): {live_rf_rate:.2f}%"
        ) / 100


    # ==========================================================================
    # LOAD DATA
    # ==========================================================================

    # Get symbols from BOTH holdings AND transactions (filter out any empty/null values)
    # CASH is handled specially - not fetched from Yahoo Finance
    holdings_symbols = [s for s in st.session_state.holdings['symbol'].unique().tolist()
                        if s and isinstance(s, str) and s.strip()]

    # Also get symbols from transactions (for stocks bought that aren't in initial holdings)
    transaction_symbols = []
    if st.session_state.transactions is not None and not st.session_state.transactions.empty:
        transaction_symbols = [s for s in st.session_state.transactions['symbol'].unique().tolist()
                               if s and isinstance(s, str) and s.strip()]

    # Combine and deduplicate
    all_symbols = list(set(holdings_symbols + transaction_symbols))

    # Separate cash from tradeable symbols
    cash_symbols = [s for s in all_symbols if s.upper() == 'CASH']
    tradeable_symbols = [s for s in all_symbols if s.upper() != 'CASH']
    symbols = all_symbols  # Keep full list for reference

    all_symbols = list(set(tradeable_symbols + [benchmark]))

    # Fetch price data
    with st.spinner("Loading market data..."):
        prices_df = fetch_price_data(all_symbols, start_date, end_date)

    # Filter symbols to only those with valid price data
    if not prices_df.empty:
        valid_symbols = [s for s in tradeable_symbols if s in prices_df.columns and prices_df[s].notna().any()]
        if len(valid_symbols) < len(tradeable_symbols):
            missing = set(tradeable_symbols) - set(valid_symbols)
            if missing:
                st.warning(f"Could not fetch price data for: {', '.join(missing)}")
        tradeable_symbols = valid_symbols

        # Add CASH as constant $1 price if present in holdings
        if cash_symbols:
            prices_df['CASH'] = 1.0

        # Combine tradeable symbols with cash
        symbols = tradeable_symbols + cash_symbols

    if prices_df.empty:
        st.error("Failed to load price data. Please check your internet connection and try again.")
        return

    # Estimate cost basis from prices if still zero
    if st.session_state.holdings['cost_basis'].sum() == 0:
        st.session_state.holdings = estimate_cost_basis_from_prices(
            st.session_state.holdings,
            prices_df[symbols] if symbols else prices_df
        )

    # Separate benchmark
    benchmark_prices = prices_df[benchmark] if benchmark in prices_df.columns else pd.Series()
    benchmark_returns = benchmark_prices.pct_change().dropna()

    # Get initial cash from session state
    initial_cash = st.session_state.get('initial_cash', 0.0)

    # Reconstruct portfolio using new function that properly tracks holdings through time
    portfolio_df, weights_df, cash_flows_df, current_holdings_df = reconstruct_portfolio_from_initial(
        st.session_state.transactions,
        st.session_state.holdings,
        prices_df[symbols],
        initial_cash
    )

    # Calculate Time-Weighted Returns (TWR)
    twr_metrics = calculate_time_weighted_return(portfolio_df, cash_flows_df)

    # Calculate Money-Weighted Returns (MWR/IRR)
    mwr_metrics = calculate_money_weighted_return(portfolio_df, cash_flows_df, initial_cash)

    # Calculate adjusted benchmark (accounts for same cash flows as portfolio)
    adjusted_benchmark = calculate_adjusted_benchmark(benchmark_prices, portfolio_df, cash_flows_df)

    # Calculate SIMPLE returns (WITH cash flow effects - what actually happened to portfolio value)
    simple_returns = calculate_returns(portfolio_df)  # Raw pct_change, includes CF impact

    # Calculate simple total/annualized return (includes cash flow effects)
    if not simple_returns.empty:
        simple_total = (1 + simple_returns).prod() - 1
        n_days_simple = len(simple_returns)
        n_years_simple = n_days_simple / 252 if n_days_simple > 0 else 1
        simple_annualized = (1 + simple_total) ** (1 / n_years_simple) - 1 if n_years_simple > 0 else simple_total
    else:
        simple_total = 0
        simple_annualized = 0

    # Get TWR-adjusted daily returns for Sharpe/Sortino calculations
    # These returns EXCLUDE the impact of external cash flows (pure investment performance)
    if 'daily_twr' in twr_metrics and not twr_metrics['daily_twr'].empty:
        # Convert cumulative TWR back to daily returns
        cumulative_twr = twr_metrics['daily_twr'] + 1  # Convert from return to growth factor
        twr_daily_returns = cumulative_twr.pct_change().dropna()
    else:
        twr_daily_returns = simple_returns

    # Keep portfolio_returns as alias for compatibility with rest of code
    portfolio_returns = twr_daily_returns

    # Calculate performance metrics using TWR-adjusted daily returns (for proper Sharpe, etc.)
    metrics = calculate_performance_metrics(portfolio_returns, benchmark_returns, risk_free_rate)

    # Add all return metrics
    # TWR = Time-Weighted Return (excludes cash flow timing, measures manager skill)
    metrics['twr_total'] = twr_metrics['twr_total']
    metrics['twr_annualized'] = twr_metrics['twr_annualized']

    # MWR = Money-Weighted Return (includes cash flow timing, measures investor experience)
    metrics['mwr_total'] = mwr_metrics['mwr_total']
    metrics['mwr_annualized'] = mwr_metrics['mwr_annualized']

    # Simple = Raw returns (includes cash flows as if they were gains/losses)
    metrics['simple_total'] = simple_total
    metrics['simple_annualized'] = simple_annualized

    # Benchmark returns - both raw and cash-flow adjusted
    # Total returns (not annualized)
    metrics['raw_benchmark_total'] = adjusted_benchmark['raw_total']  # Raw benchmark total return
    metrics['adjusted_benchmark_total'] = adjusted_benchmark['adjusted_total']  # Cash-flow adjusted total return
    # Annualized returns
    metrics['raw_benchmark_return'] = adjusted_benchmark['raw_annualized']  # Raw benchmark annualized
    metrics['adjusted_benchmark_return'] = adjusted_benchmark['adjusted_annualized']  # Cash-flow adjusted annualized
    metrics['benchmark_return'] = adjusted_benchmark['raw_annualized']  # For backward compatibility

    # Calculate Beta excluding cash (securities-only beta)
    if 'securities_value' in portfolio_df.columns and not benchmark_returns.empty:
        securities_values = portfolio_df['securities_value'].dropna()
        if len(securities_values) > 1:
            securities_returns = securities_values.pct_change().dropna()
            # Align with benchmark
            aligned_sec = pd.concat([securities_returns, benchmark_returns], axis=1, join='inner')
            if len(aligned_sec) > 10:
                aligned_sec.columns = ['securities', 'benchmark']
                cov_sec = np.cov(aligned_sec['securities'].values, aligned_sec['benchmark'].values)[0, 1]
                var_bench = aligned_sec['benchmark'].var()
                beta_ex_cash = cov_sec / var_bench if var_bench > 0 else 1
                metrics['beta_ex_cash'] = beta_ex_cash
            else:
                metrics['beta_ex_cash'] = metrics.get('beta', 1)
        else:
            metrics['beta_ex_cash'] = metrics.get('beta', 1)
    else:
        metrics['beta_ex_cash'] = metrics.get('beta', 1)

    # Update symbols list to include any new holdings from transactions (like LULU)
    current_symbols = current_holdings_df['symbol'].tolist() if not current_holdings_df.empty else symbols

    # Fetch any missing price data for new symbols
    missing_symbols = [s for s in current_symbols if s not in prices_df.columns and s.upper() != 'CASH']
    if missing_symbols:
        try:
            new_prices = fetch_price_data(missing_symbols, start_date, end_date)
            if not new_prices.empty:
                for col in new_prices.columns:
                    prices_df[col] = new_prices[col]
        except:
            pass  # Continue with available data

    # Calculate cost basis for current holdings from transactions
    if not current_holdings_df.empty:
        current_holdings_df = calculate_cost_basis_from_transactions(
            st.session_state.transactions,
            current_holdings_df
        )
        # For tickers without transactions, use start-of-range price as cost basis
        current_holdings_df = estimate_cost_basis_from_prices(
            current_holdings_df,
            prices_df
        )

    # Calculate position metrics using CURRENT holdings (after all transactions)
    position_metrics = calculate_position_metrics(
        current_holdings_df if not current_holdings_df.empty else st.session_state.holdings,
        prices_df,
        benchmark_returns
    )

    # ==========================================================================
    # MAIN TABS
    # ==========================================================================

    tabs = st.tabs([
        "📊 Overview",
        "💼 Positions & Valuation",
        "📝 Transactions",
        "⚠️ Risk Analytics",
        "🔄 Stress & Rebalance",
        "📑 Reports"
    ])

    # ==========================================================================
    # TAB 1: OVERVIEW
    # ==========================================================================
    with tabs[0]:
        st.markdown("## Portfolio Overview")

        # Extract metrics
        total_value = position_metrics['Market Value'].sum() if not position_metrics.empty else 0
        total_cost = (position_metrics['Cost Basis'] * position_metrics['Quantity']).sum() if not position_metrics.empty else 0
        total_pnl = total_value - total_cost

        # Total returns (since inception, not annualized)
        twr_total = metrics.get('twr_total', 0)
        adj_bench_total = metrics.get('adjusted_benchmark_total', 0)

        # Annualized returns
        twr_ann = metrics.get('twr_annualized', 0)
        adj_bench_ann = metrics.get('adjusted_benchmark_return', 0)

        beta_ex_cash = metrics.get('beta_ex_cash', 1)

        # KPI Cards Row 1 - Returns (Total, Since Inception)
        st.markdown("##### Returns Since Inception")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Portfolio Value",
                f"${total_value:,.0f}",
                f"P&L: ${total_pnl:,.0f}" if total_cost > 0 else "N/A"
            )

        with col2:
            st.metric(
                "Portfolio Return (Total)",
                f"{twr_total:.1%}",
                f"vs Benchmark: {(twr_total - adj_bench_total):+.1%}"
            )

        with col3:
            st.metric(
                f"{benchmark} Return (Total)",
                f"{adj_bench_total:.1%}",
                "Cash-flow adjusted"
            )

        with col4:
            st.metric(
                "Beta (Ex-Cash)",
                f"{beta_ex_cash:.2f}",
                f"vs {benchmark}"
            )

        # KPI Cards Row 2 - Annualized & Risk Metrics
        st.markdown("##### Annualized & Risk Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "TWR (Annualized)",
                f"{twr_ann:.1%}",
                f"vs Bench: {(twr_ann - adj_bench_ann):+.1%}"
            )

        with col2:
            st.metric(
                "Volatility (Ann.)",
                f"{metrics.get('volatility', 0):.1%}",
                f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}"
            )

        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0):.1%}",
                f"Calmar: {metrics.get('calmar_ratio', 0):.2f}"
            )

        with col4:
            st.metric(
                "Sortino Ratio",
                f"{metrics.get('sortino_ratio', 0):.2f}",
                "Downside risk"
            )

        # Compact explanation
        with st.expander("Metrics Explained"):
            st.markdown("""
            - **Total Return**: Cumulative return since inception (not annualized)
            - **TWR (Annualized)**: Time-Weighted Return - manager performance excluding cash flow timing
            - **Adjusted Benchmark**: Benchmark return with same cash flows as your portfolio
            - **Beta (Ex-Cash)**: Market sensitivity of securities only (excludes cash drag)
            """)

        st.markdown("---")

        # Performance Chart
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Cumulative Performance (Cash-Flow Adjusted)")

            if not portfolio_returns.empty:
                # Calculate cumulative returns
                cum_portfolio = (1 + portfolio_returns).cumprod()

                # Use adjusted benchmark values (already accounts for same cash flows)
                adj_bench_values = adjusted_benchmark.get('adjusted_benchmark_values', pd.Series())
                if not adj_bench_values.empty:
                    # Normalize to start at 1
                    cum_adj_benchmark = adj_bench_values / adj_bench_values.iloc[0]
                else:
                    cum_adj_benchmark = (1 + benchmark_returns).cumprod()

                # Also show raw benchmark for comparison
                cum_raw_benchmark = (1 + benchmark_returns).cumprod()

                # Align indices
                aligned = pd.concat([cum_portfolio, cum_adj_benchmark, cum_raw_benchmark], axis=1, join='inner')
                aligned.columns = ['Portfolio', f'{benchmark} (Adj.)', f'{benchmark} (Raw)']

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=aligned.index,
                    y=aligned['Portfolio'],
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='#667eea', width=2)
                ))

                fig.add_trace(go.Scatter(
                    x=aligned.index,
                    y=aligned[f'{benchmark} (Adj.)'],
                    mode='lines',
                    name=f'{benchmark} (Cash-Flow Adjusted)',
                    line=dict(color='#22c55e', width=2)
                ))

                fig.add_trace(go.Scatter(
                    x=aligned.index,
                    y=aligned[f'{benchmark} (Raw)'],
                    mode='lines',
                    name=f'{benchmark} (Raw)',
                    line=dict(color='#8b8fa3', width=2, dash='dash')
                ))

                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    ),
                    xaxis=dict(gridcolor='#2d3250'),
                    yaxis=dict(gridcolor='#2d3250', tickformat='.1%')
                )

                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Allocation")

            if not position_metrics.empty:
                fig = px.pie(
                    position_metrics,
                    values='Market Value',
                    names='Symbol',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )

                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=True,
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=-0.2
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

        # Drawdown chart
        st.markdown("### Drawdown Analysis")

        if not portfolio_returns.empty:
            cumulative = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values,
                fill='tozeroy',
                mode='lines',
                name='Drawdown',
                line=dict(color='#ef4444', width=1),
                fillcolor='rgba(239, 68, 68, 0.3)'
            ))

            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(gridcolor='#2d3250'),
                yaxis=dict(gridcolor='#2d3250', tickformat='.1%')
            )

            st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # TAB 2: POSITIONS & VALUATION
    # ==========================================================================
    with tabs[1]:
        st.markdown("## Positions & Valuation")

        # Editable holdings table
        st.markdown("### Current Holdings")
        st.caption("Edit values directly in the table below")

        if not position_metrics.empty:
            # Format for display
            display_df = position_metrics.copy()

            edited_df = st.data_editor(
                display_df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Quantity": st.column_config.NumberColumn("Qty", format="%d"),
                    "Cost Basis": st.column_config.NumberColumn("Cost Basis", format="$%.2f"),
                    "Current Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "Market Value": st.column_config.NumberColumn("Mkt Value", format="$%,.0f"),
                    "Unrealized P&L": st.column_config.NumberColumn("Unreal. P&L", format="$%,.0f"),
                    "P&L %": st.column_config.NumberColumn("P&L %", format="%.1f%%"),
                    "Weight": st.column_config.NumberColumn("Weight", format="%.1f%%"),
                    "Volatility": st.column_config.NumberColumn("Vol", format="%.1%"),
                    "Beta": st.column_config.NumberColumn("Beta", format="%.2f"),
                    "Country": st.column_config.TextColumn("Country", width="small"),
                    "CRP (%)": st.column_config.NumberColumn("CRP", format="%.1f%%"),
                }
            )

        st.markdown("---")

        # Position-level charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### P&L by Position")
            if not position_metrics.empty:
                fig = px.bar(
                    position_metrics.sort_values('Unrealized P&L'),
                    x='Symbol',
                    y='Unrealized P&L',
                    color='Unrealized P&L',
                    color_continuous_scale=['#ef4444', '#22c55e']
                )

                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(gridcolor='#2d3250'),
                    yaxis=dict(gridcolor='#2d3250'),
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Weight vs Position Limit")
            if not position_metrics.empty:
                fig = go.Figure()

                colors = ['#ef4444' if w > position_limit else '#667eea'
                          for w in position_metrics['Weight']]

                fig.add_trace(go.Bar(
                    x=position_metrics['Symbol'],
                    y=position_metrics['Weight'],
                    marker_color=colors,
                    name='Weight'
                ))

                fig.add_hline(
                    y=position_limit,
                    line_dash="dash",
                    line_color="#fbbf24",
                    annotation_text=f"Limit: {position_limit}%"
                )

                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(gridcolor='#2d3250'),
                    yaxis=dict(gridcolor='#2d3250', title='Weight %'),
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

        # Drift alerts
        if not position_metrics.empty:
            over_limit = position_metrics[position_metrics['Weight'] > position_limit]
            if not over_limit.empty:
                st.warning(f"⚠️ **Position Limit Breaches:** {', '.join(over_limit['Symbol'].tolist())} exceed the {position_limit}% limit")

        # Country Risk Premium analysis
        if enable_crp:
            st.markdown("---")
            st.markdown("### Country Risk Premium Analysis")

            with st.expander("ℹ️ About Country Risk Premiums", expanded=False):
                st.markdown("""
                Country Risk Premiums (CRP) represent the additional return required to compensate
                for investing in riskier countries. These estimates are based on Damodaran's methodology
                and consider factors like sovereign credit ratings, currency volatility, and political stability.

                **Adjustment Method:** Returns are adjusted by adding the weighted CRP to the risk-free rate
                in CAPM calculations.
                """)

            if not position_metrics.empty:
                crp_summary = position_metrics.groupby('Country').agg({
                    'Market Value': 'sum',
                    'CRP (%)': 'first'
                }).reset_index()
                crp_summary['Weight'] = crp_summary['Market Value'] / crp_summary['Market Value'].sum() * 100

                weighted_crp = (crp_summary['Weight'] * crp_summary['CRP (%)']).sum() / 100

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric("Portfolio Weighted CRP", f"{weighted_crp:.2f}%")
                    st.dataframe(crp_summary, use_container_width=True, hide_index=True)

                with col2:
                    fig = px.treemap(
                        position_metrics,
                        path=['Country', 'Symbol'],
                        values='Market Value',
                        color='CRP (%)',
                        color_continuous_scale='RdYlGn_r'
                    )

                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=350,
                        margin=dict(l=10, r=10, t=10, b=10)
                    )

                    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # ==========================================================================
    # TAB 3: TRANSACTIONS
    # ==========================================================================
    with tabs[2]:
        st.markdown("## Transaction History")

        if st.session_state.transactions is not None and not st.session_state.transactions.empty:
            txn_df = st.session_state.transactions.copy()

            # Summary metrics - Row 1
            col1, col2, col3, col4 = st.columns(4)

            total_buys = len(txn_df[txn_df['side'] == 'BUY'])
            total_sells = len(txn_df[txn_df['side'] == 'SELL'])
            total_withdrawals = len(txn_df[txn_df['side'].isin(['WITHDRAWAL', 'REBALANCE'])])
            total_deposits = len(txn_df[txn_df['side'] == 'DEPOSIT'])
            total_fees = txn_df['fees'].sum() if 'fees' in txn_df.columns else 0

            # Calculate total buy/sell values
            buy_txns = txn_df[txn_df['side'] == 'BUY']
            sell_txns = txn_df[txn_df['side'] == 'SELL']
            withdrawal_txns = txn_df[txn_df['side'].isin(['WITHDRAWAL', 'REBALANCE'])]
            deposit_txns = txn_df[txn_df['side'] == 'DEPOSIT']

            total_buy_value = (buy_txns['quantity'] * buy_txns['price']).sum() if not buy_txns.empty else 0
            total_sell_value = (sell_txns['quantity'] * sell_txns['price']).sum() if not sell_txns.empty else 0
            total_withdrawal_value = (withdrawal_txns['quantity'] * withdrawal_txns['price'].fillna(1)).sum() if not withdrawal_txns.empty else 0
            total_deposit_value = (deposit_txns['quantity'] * deposit_txns['price'].fillna(1)).sum() if not deposit_txns.empty else 0

            with col1:
                st.metric("Total Transactions", len(txn_df))
            with col2:
                st.metric("Buy Orders", total_buys, f"${total_buy_value:,.0f}")
            with col3:
                st.metric("Sell Orders", total_sells, f"${total_sell_value:,.0f}")
            with col4:
                st.metric("Total Fees", f"${total_fees:,.2f}")

            # Summary metrics - Row 2 (Cash movements)
            if total_withdrawals > 0 or total_deposits > 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Withdrawals/Rebalance", total_withdrawals, f"-${total_withdrawal_value:,.0f}", delta_color="inverse")
                with col2:
                    st.metric("Deposits", total_deposits, f"+${total_deposit_value:,.0f}")
                with col3:
                    net_cash_movement = total_deposit_value - total_withdrawal_value
                    st.metric("Net Cash Movement", f"${net_cash_movement:,.0f}")
                with col4:
                    pass  # Empty column for alignment

            st.markdown("---")

            # Transaction filters
            col1, col2, col3 = st.columns(3)

            # Get all unique transaction types in the data
            all_sides = txn_df['side'].unique().tolist()
            default_sides = [s for s in all_sides if s in ['BUY', 'SELL', 'WITHDRAWAL', 'REBALANCE', 'DEPOSIT']]

            with col1:
                side_filter = st.multiselect(
                    "Filter by Action",
                    options=all_sides,
                    default=default_sides if default_sides else all_sides
                )

            with col2:
                symbols_in_txns = txn_df['symbol'].unique().tolist()
                symbol_filter = st.multiselect(
                    "Filter by Symbol",
                    options=symbols_in_txns,
                    default=symbols_in_txns
                )

            with col3:
                sort_order = st.selectbox(
                    "Sort by Date",
                    options=['Newest First', 'Oldest First'],
                    index=0
                )

            # Apply filters
            filtered_txns = txn_df[
                (txn_df['side'].isin(side_filter)) &
                (txn_df['symbol'].isin(symbol_filter))
            ].copy()

            # Sort
            ascending = sort_order == 'Oldest First'
            filtered_txns = filtered_txns.sort_values('date', ascending=ascending)

            # Calculate transaction value (handle NaN prices for cash transactions)
            filtered_txns['price'] = filtered_txns['price'].fillna(1)
            filtered_txns['fees'] = filtered_txns['fees'].fillna(0)
            filtered_txns['value'] = filtered_txns['quantity'] * filtered_txns['price']

            # Display table
            st.markdown("### All Transactions")

            display_txns = filtered_txns.copy()
            display_txns['date'] = pd.to_datetime(display_txns['date']).dt.strftime('%Y-%m-%d')

            st.dataframe(
                display_txns[['date', 'symbol', 'side', 'quantity', 'price', 'fees', 'value']],
                use_container_width=True,
                column_config={
                    "date": st.column_config.TextColumn("Date"),
                    "symbol": st.column_config.TextColumn("Symbol"),
                    "side": st.column_config.TextColumn("Action"),
                    "quantity": st.column_config.NumberColumn("Shares/Amount", format="%.2f"),
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "fees": st.column_config.NumberColumn("Fees", format="$%.2f"),
                    "value": st.column_config.NumberColumn("Value", format="$%,.2f"),
                },
                hide_index=True
            )

            st.markdown("---")

            # Transaction charts
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Transactions by Symbol")
                txn_by_symbol = txn_df.groupby(['symbol', 'side']).agg({
                    'quantity': 'sum',
                    'price': 'mean'
                }).reset_index()
                txn_by_symbol['price'] = txn_by_symbol['price'].fillna(1)
                txn_by_symbol['value'] = txn_by_symbol['quantity'] * txn_by_symbol['price']

                fig = px.bar(
                    txn_by_symbol,
                    x='symbol',
                    y='value',
                    color='side',
                    barmode='group',
                    color_discrete_map={
                        'BUY': '#22c55e',
                        'SELL': '#ef4444',
                        'WITHDRAWAL': '#f59e0b',
                        'REBALANCE': '#8b5cf6',
                        'DEPOSIT': '#06b6d4'
                    }
                )
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(gridcolor='#2d3250'),
                    yaxis=dict(gridcolor='#2d3250', title='Value ($)'),
                    legend=dict(title='Action')
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Transaction Timeline")
                txn_timeline = txn_df.copy()
                txn_timeline['value'] = txn_timeline['quantity'] * txn_timeline['price'].fillna(1)
                txn_timeline['value'] = txn_timeline.apply(
                    lambda x: x['value'] if x['side'] in ['BUY', 'DEPOSIT'] else -x['value'], axis=1
                )

                fig = px.scatter(
                    txn_timeline,
                    x='date',
                    y='value',
                    color='side',
                    size=abs(txn_timeline['value']),
                    hover_data=['symbol', 'quantity', 'price'],
                    color_discrete_map={
                        'BUY': '#22c55e',
                        'SELL': '#ef4444',
                        'WITHDRAWAL': '#f59e0b',
                        'REBALANCE': '#8b5cf6',
                        'DEPOSIT': '#06b6d4'
                    }
                )
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(gridcolor='#2d3250', title='Date'),
                    yaxis=dict(gridcolor='#2d3250', title='Value ($)'),
                    legend=dict(title='Action')
                )
                st.plotly_chart(fig, use_container_width=True)

            # Cash flow summary
            st.markdown("---")
            st.markdown("### Cash Flow Summary")

            # Calculate cash flows by month
            txn_cashflow = txn_df.copy()
            txn_cashflow['date'] = pd.to_datetime(txn_cashflow['date'])
            txn_cashflow['month'] = txn_cashflow['date'].dt.to_period('M')

            def calc_cash_flow(row):
                qty = float(row.get('quantity', 0) or 0)
                price = float(row.get('price', 0) or 1)
                fees = float(row.get('fees', 0) or 0)
                side = str(row.get('side', '')).upper()

                if side == 'SELL':
                    return qty * price - fees
                elif side == 'BUY':
                    return -(qty * price + fees)
                elif side in ['WITHDRAWAL', 'REBALANCE']:
                    return -(qty * price)  # Cash leaving portfolio
                elif side == 'DEPOSIT':
                    return qty * price  # Cash entering portfolio
                return 0

            txn_cashflow['cash_flow'] = txn_cashflow.apply(calc_cash_flow, axis=1)

            monthly_cf = txn_cashflow.groupby('month')['cash_flow'].sum().reset_index()
            monthly_cf['month'] = monthly_cf['month'].astype(str)

            fig = px.bar(
                monthly_cf,
                x='month',
                y='cash_flow',
                color='cash_flow',
                color_continuous_scale=['#ef4444', '#fbbf24', '#22c55e']
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(gridcolor='#2d3250', title='Month'),
                yaxis=dict(gridcolor='#2d3250', title='Net Cash Flow ($)'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Current cash balance (initial_cash from holdings file)
            initial_cash = st.session_state.get('initial_cash', 0.0)
            current_cash, total_withdrawals, total_deposits = calculate_cash_from_transactions(txn_df, initial_cash)

            # Display cash summary
            st.markdown("---")
            st.markdown("### Cash Balance Summary")
            cash_col1, cash_col2, cash_col3, cash_col4 = st.columns(4)
            with cash_col1:
                st.metric("Starting Cash", f"${initial_cash:,.2f}")
            with cash_col2:
                st.metric("Total Withdrawals", f"${total_withdrawals:,.2f}", delta=f"-${total_withdrawals:,.2f}" if total_withdrawals > 0 else None, delta_color="inverse")
            with cash_col3:
                st.metric("Total Deposits", f"${total_deposits:,.2f}", delta=f"+${total_deposits:,.2f}" if total_deposits > 0 else None)
            with cash_col4:
                st.metric("Current Cash", f"${current_cash:,.2f}")

        else:
            st.warning("No transactions loaded. Upload a transactions CSV or check your data files.")

    # ==========================================================================
    # TAB 4: RISK ANALYTICS
    # ==========================================================================
    with tabs[3]:
        st.markdown("## Risk Analytics")

        # VaR/CVaR section
        col1, col2, col3 = st.columns(3)

        # Calculate VaR metrics
        hist_var, hist_cvar = calculate_var_cvar(portfolio_returns, var_confidence, 'historical')
        param_var, param_cvar = calculate_var_cvar(portfolio_returns, var_confidence, 'parametric')

        with col1:
            st.markdown("### Historical VaR")
            st.metric(
                f"1-Day VaR ({var_confidence:.0%})",
                f"${abs(hist_var * total_value):,.0f}",
                f"{hist_var:.2%} of portfolio"
            )
            st.metric(
                f"CVaR (Expected Shortfall)",
                f"${abs(hist_cvar * total_value):,.0f}",
                f"{hist_cvar:.2%} of portfolio"
            )

        with col2:
            st.markdown("### Parametric VaR")
            st.metric(
                f"1-Day VaR ({var_confidence:.0%})",
                f"${abs(param_var * total_value):,.0f}",
                f"{param_var:.2%} of portfolio"
            )
            st.metric(
                f"CVaR (Expected Shortfall)",
                f"${abs(param_cvar * total_value):,.0f}",
                f"{param_cvar:.2%} of portfolio"
            )

        with col3:
            st.markdown("### Risk Metrics")
            st.metric("Annualized Volatility", f"{metrics.get('volatility', 0):.1%}")
            st.metric("Skewness", f"{metrics.get('skewness', 0):.2f}")
            st.metric("Kurtosis", f"{metrics.get('kurtosis', 0):.2f}")

        st.markdown("---")

        # Monte Carlo VaR
        st.markdown("### Monte Carlo Simulation")

        col1, col2 = st.columns([2, 1])

        with col1:
            mc_days = st.slider("Simulation Horizon (days)", 21, 252, 63)
            mc_sims = st.selectbox("Number of Simulations", [1000, 5000, 10000], index=1)

        if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                simulations, final_values = monte_carlo_simulation(
                    portfolio_returns,
                    n_simulations=mc_sims,
                    n_days=mc_days,
                    initial_value=total_value
                )

                if len(simulations) > 0:
                    # MC VaR
                    mc_var = np.percentile(final_values, (1 - var_confidence) * 100)
                    mc_var_pct = (mc_var - total_value) / total_value
                    mc_cvar = final_values[final_values <= mc_var].mean()
                    mc_cvar_pct = (mc_cvar - total_value) / total_value

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            f"MC VaR ({mc_days}d, {var_confidence:.0%})",
                            f"${total_value - mc_var:,.0f}",
                            f"{mc_var_pct:.1%}"
                        )

                    with col2:
                        st.metric(
                            f"MC CVaR ({mc_days}d)",
                            f"${total_value - mc_cvar:,.0f}",
                            f"{mc_cvar_pct:.1%}"
                        )

                    with col3:
                        median_value = np.median(final_values)
                        st.metric(
                            "Median Outcome",
                            f"${median_value:,.0f}",
                            f"{(median_value - total_value) / total_value:.1%}"
                        )

                    # Fan chart
                    st.markdown("#### Simulation Fan Chart")

                    percentiles = [5, 25, 50, 75, 95]
                    percentile_values = np.percentile(simulations, percentiles, axis=0)

                    fig = go.Figure()

                    # Add percentile bands
                    x_range = list(range(mc_days))

                    fig.add_trace(go.Scatter(
                        x=x_range + x_range[::-1],
                        y=list(percentile_values[0]) + list(percentile_values[4])[::-1],
                        fill='toself',
                        fillcolor='rgba(102, 126, 234, 0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='5th-95th %ile'
                    ))

                    fig.add_trace(go.Scatter(
                        x=x_range + x_range[::-1],
                        y=list(percentile_values[1]) + list(percentile_values[3])[::-1],
                        fill='toself',
                        fillcolor='rgba(102, 126, 234, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='25th-75th %ile'
                    ))

                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=percentile_values[2],
                        mode='lines',
                        name='Median',
                        line=dict(color='#667eea', width=2)
                    ))

                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis=dict(gridcolor='#2d3250', title='Days'),
                        yaxis=dict(gridcolor='#2d3250', title='Portfolio Value', tickformat='$,.0f'),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Distribution of final values
                    st.markdown("#### Distribution of Final Values")

                    fig = go.Figure()

                    fig.add_trace(go.Histogram(
                        x=final_values,
                        nbinsx=50,
                        marker_color='#667eea',
                        opacity=0.7
                    ))

                    # Add VaR line
                    fig.add_vline(
                        x=mc_var,
                        line_dash="dash",
                        line_color="#ef4444",
                        annotation_text=f"VaR: ${mc_var:,.0f}"
                    )

                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=300,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis=dict(gridcolor='#2d3250', title='Portfolio Value', tickformat='$,.0f'),
                        yaxis=dict(gridcolor='#2d3250', title='Frequency'),
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Correlation matrix
        st.markdown("### Correlation Matrix")

        if symbols and len(symbols) > 1:
            corr_matrix = calculate_correlation_matrix(prices_df[symbols])

            if not corr_matrix.empty:
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    zmin=-1,
                    zmax=1
                )

                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=20)
                )

                st.plotly_chart(fig, use_container_width=True)

        # Rolling metrics
        st.markdown("---")
        st.markdown("### Rolling Risk Metrics")

        rolling_window = st.slider("Rolling Window (days)", 21, 126, 63)

        if not portfolio_returns.empty:
            rolling_vol = portfolio_returns.rolling(rolling_window).std() * np.sqrt(252)
            rolling_sharpe = (portfolio_returns.rolling(rolling_window).mean() * 252 - risk_free_rate) / rolling_vol

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

            fig.add_trace(
                go.Scatter(x=rolling_vol.index, y=rolling_vol, name='Rolling Vol',
                          line=dict(color='#667eea')),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name='Rolling Sharpe',
                          line=dict(color='#22c55e')),
                row=2, col=1
            )

            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )

            fig.update_xaxes(gridcolor='#2d3250')
            fig.update_yaxes(gridcolor='#2d3250')

            st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # TAB 5: STRESS & REBALANCE
    # ==========================================================================
    with tabs[4]:
        st.markdown("## Stress Testing & Rebalancing")

        # Stress Testing
        st.markdown("### 📉 Stress Testing")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Custom Scenario**")

            custom_shock = st.slider("Market-wide shock (%)", -60, 20, -20)
            tech_shock = st.slider("Tech sector shock (%)", -60, 20, -25)

            custom_scenario = {
                'Custom Scenario': {
                    'market': custom_shock / 100,
                    'tech': tech_shock / 100,
                }
            }

        if st.button("🔥 Run Stress Tests", type="primary"):
            with st.spinner("Running stress scenarios..."):
                stress_results = run_stress_tests(
                    st.session_state.holdings,
                    prices_df[symbols]
                )

                # Add custom scenario
                custom_results = run_stress_tests(
                    st.session_state.holdings,
                    prices_df[symbols],
                    custom_scenario
                )

                all_results = pd.concat([stress_results, custom_results], ignore_index=True)

                with col2:
                    st.markdown("**Stress Test Results**")

                    fig = px.bar(
                        all_results,
                        y='Scenario',
                        x='Portfolio Loss (%)',
                        orientation='h',
                        color='Portfolio Loss (%)',
                        color_continuous_scale=['#ef4444', '#fbbf24', '#22c55e']
                    )

                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=350,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis=dict(gridcolor='#2d3250', title='Loss %'),
                        yaxis=dict(gridcolor='#2d3250'),
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Results table
                st.dataframe(
                    all_results.style.format({
                        'Portfolio Loss ($)': '${:,.0f}',
                        'Portfolio Loss (%)': '{:.1f}%'
                    }).background_gradient(
                        subset=['Portfolio Loss (%)'],
                        cmap='RdYlGn_r'
                    ),
                    use_container_width=True,
                    hide_index=True
                )

        st.markdown("---")

        # Rebalancing
        st.markdown("### ⚖️ Portfolio Rebalancing")

        col1, col2 = st.columns(2)

        with col1:
            rebalance_method = st.selectbox(
                "Optimization Method",
                ['max_sharpe', 'min_variance', 'risk_parity'],
                format_func=lambda x: {
                    'max_sharpe': 'Maximum Sharpe Ratio',
                    'min_variance': 'Minimum Variance',
                    'risk_parity': 'Risk Parity'
                }[x]
            )

            max_weight = st.slider("Max Weight per Position (%)", 10, 50, 25)
            min_weight = st.slider("Min Weight per Position (%)", 0, 10, 2)

        if st.button("⚖️ Optimize Portfolio", type="primary"):
            with st.spinner("Running optimization..."):
                # Get returns for optimization
                returns_df = prices_df[symbols].pct_change().dropna()

                optimization_result = optimize_portfolio(
                    returns_df,
                    method=rebalance_method,
                    risk_free_rate=risk_free_rate,
                    max_weight=max_weight / 100,
                    min_weight=min_weight / 100
                )

                if optimization_result.get('success'):
                    with col2:
                        st.markdown("**Optimal Portfolio Metrics**")

                        st.metric(
                            "Expected Return",
                            f"{optimization_result['expected_return']:.1%}"
                        )
                        st.metric(
                            "Expected Volatility",
                            f"{optimization_result['expected_volatility']:.1%}"
                        )
                        st.metric(
                            "Sharpe Ratio",
                            f"{optimization_result['sharpe_ratio']:.2f}"
                        )

                    st.markdown("#### Optimal Weights")

                    weights_df = pd.DataFrame([
                        {'Symbol': s, 'Optimal Weight': w * 100}
                        for s, w in optimization_result['weights'].items()
                    ])

                    # Compare with current
                    current_weights = position_metrics[['Symbol', 'Weight']].copy()
                    current_weights.columns = ['Symbol', 'Current Weight']

                    comparison = weights_df.merge(current_weights, on='Symbol', how='outer').fillna(0)
                    comparison['Change'] = comparison['Optimal Weight'] - comparison['Current Weight']

                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        name='Current',
                        x=comparison['Symbol'],
                        y=comparison['Current Weight'],
                        marker_color='#8b8fa3'
                    ))

                    fig.add_trace(go.Bar(
                        name='Optimal',
                        x=comparison['Symbol'],
                        y=comparison['Optimal Weight'],
                        marker_color='#667eea'
                    ))

                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=350,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis=dict(gridcolor='#2d3250'),
                        yaxis=dict(gridcolor='#2d3250', title='Weight %'),
                        barmode='group',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Generate rebalance transactions
                    st.markdown("#### Rebalancing Transactions")

                    current_holdings = st.session_state.holdings.set_index('symbol')['quantity'].to_dict()

                    rebalance_txns = generate_rebalance_transactions(
                        current_holdings,
                        optimization_result['weights'],
                        prices_df[symbols],
                        total_value
                    )

                    if not rebalance_txns.empty:
                        st.dataframe(
                            rebalance_txns.style.format({
                                'Est. Price': '${:.2f}',
                                'Est. Value': '${:,.0f}',
                                'Current Weight': '{:.1f}%',
                                'Target Weight': '{:.1f}%'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )

                        # Export button
                        rebalance_csv = rebalance_txns.to_csv(index=False)
                        st.download_button(
                            "📥 Download Rebalance Orders",
                            rebalance_csv,
                            file_name="rebalance_orders.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("Portfolio is already optimally balanced!")
                else:
                    st.error("Optimization failed. Try adjusting constraints.")

        # Efficient Frontier
        st.markdown("---")
        st.markdown("### Efficient Frontier")

        if st.button("📈 Generate Efficient Frontier"):
            with st.spinner("Calculating efficient frontier..."):
                returns_df = prices_df[symbols].pct_change().dropna()
                mean_returns = returns_df.mean() * 252
                cov_matrix = returns_df.cov() * 252

                # Generate frontier points
                n_portfolios = 100
                results = []

                target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_portfolios)

                for target in target_returns:
                    try:
                        result = optimize_portfolio(
                            returns_df,
                            method='min_variance',
                            target_return=target,
                            max_weight=0.5,
                            min_weight=0.0
                        )
                        if result.get('success'):
                            results.append({
                                'Return': result['expected_return'],
                                'Volatility': result['expected_volatility'],
                                'Sharpe': result['sharpe_ratio']
                            })
                    except:
                        continue

                if results:
                    frontier_df = pd.DataFrame(results)

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=frontier_df['Volatility'],
                        y=frontier_df['Return'],
                        mode='lines',
                        name='Efficient Frontier',
                        line=dict(color='#667eea', width=3)
                    ))

                    # Add current portfolio
                    fig.add_trace(go.Scatter(
                        x=[metrics.get('volatility', 0)],
                        y=[metrics.get('annualized_return', 0)],
                        mode='markers',
                        name='Current Portfolio',
                        marker=dict(color='#fbbf24', size=15, symbol='star')
                    ))

                    # Add individual assets
                    for symbol in symbols:
                        asset_returns = returns_df[symbol]
                        asset_vol = asset_returns.std() * np.sqrt(252)
                        asset_ret = asset_returns.mean() * 252

                        fig.add_trace(go.Scatter(
                            x=[asset_vol],
                            y=[asset_ret],
                            mode='markers+text',
                            name=symbol,
                            text=[symbol],
                            textposition='top center',
                            marker=dict(size=10)
                        ))

                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=500,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis=dict(gridcolor='#2d3250', title='Volatility', tickformat='.1%'),
                        yaxis=dict(gridcolor='#2d3250', title='Return', tickformat='.1%'),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02)
                    )

                    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # TAB 6: REPORTS
    # ==========================================================================
    with tabs[5]:
        st.markdown("## Reports & Export")

        # Summary metrics tables
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Return Metrics")

            return_data = {
                'Metric': [
                    'TWR (Total)',
                    'TWR (Annualized)',
                    'MWR/IRR (Total)',
                    'MWR/IRR (Annualized)',
                    'Simple Return (Total)',
                    'Simple Return (Annualized)',
                    f'{benchmark} Return (Ann.)',
                    'Timing Impact (MWR - TWR)',
                ],
                'Value': [
                    f"{metrics.get('twr_total', 0):.2%}",
                    f"{metrics.get('twr_annualized', 0):.2%}",
                    f"{metrics.get('mwr_total', 0):.2%}",
                    f"{metrics.get('mwr_annualized', 0):.2%}",
                    f"{metrics.get('simple_total', 0):.2%}",
                    f"{metrics.get('simple_annualized', 0):.2%}",
                    f"{metrics.get('raw_benchmark_return', 0):.2%}",
                    f"{(metrics.get('mwr_annualized', 0) - metrics.get('twr_annualized', 0)):.2%}",
                ],
                'Description': [
                    'Manager performance (excl. cash flows)',
                    'Annualized TWR',
                    'Investor experience (incl. timing)',
                    'Annualized MWR',
                    'Raw value change (incl. cash flows)',
                    'Annualized simple return',
                    'Benchmark price return',
                    'Positive = good timing',
                ]
            }

            st.dataframe(pd.DataFrame(return_data), use_container_width=True, hide_index=True)

        with col2:
            st.markdown("### Risk-Adjusted Metrics")

            risk_adj_data = {
                'Metric': [
                    'Annualized Volatility',
                    'Sharpe Ratio',
                    'Sortino Ratio',
                    'Calmar Ratio',
                    'Maximum Drawdown',
                    'Beta (Total)',
                    'Beta (Ex-Cash)',
                    'Alpha (Jensen\'s)',
                    'R-Squared',
                    'Tracking Error',
                    'Information Ratio',
                ],
                'Value': [
                    f"{metrics.get('volatility', 0):.2%}",
                    f"{metrics.get('sharpe_ratio', 0):.2f}",
                    f"{metrics.get('sortino_ratio', 0):.2f}",
                    f"{metrics.get('calmar_ratio', 0):.2f}",
                    f"{metrics.get('max_drawdown', 0):.2%}",
                    f"{metrics.get('beta', 1):.2f}",
                    f"{metrics.get('beta_ex_cash', 1):.2f}",
                    f"{metrics.get('alpha', 0):.2%}",
                    f"{metrics.get('r_squared', 0):.2%}",
                    f"{metrics.get('tracking_error', 0):.2%}",
                    f"{metrics.get('information_ratio', 0):.2f}",
                ],
                'Description': [
                    'Standard deviation (annualized)',
                    '(Return - Rf) / Volatility',
                    'Return / Downside deviation',
                    'Return / Max Drawdown',
                    'Peak to trough decline',
                    'Market sensitivity (incl. cash)',
                    'Market sensitivity (excl. cash)',
                    'Excess return vs CAPM',
                    'Variance explained by benchmark',
                    'Std dev of excess returns',
                    'Alpha / Tracking Error',
                ]
            }

            st.dataframe(pd.DataFrame(risk_adj_data), use_container_width=True, hide_index=True)

        st.markdown("---")

        # Additional risk metrics
        st.markdown("### Additional Risk Metrics")
        col1, col2 = st.columns(2)

        with col1:
            risk_data = {
                'Metric': [
                    f'VaR ({var_confidence:.0%}, 1-day)',
                    'CVaR (Expected Shortfall)',
                    'Positive Days %',
                    'Best Day',
                    'Worst Day',
                    'Skewness',
                    'Kurtosis'
                ],
                'Value': [
                    f"{hist_var:.2%}",
                    f"{hist_cvar:.2%}",
                    f"{metrics.get('positive_days', 0):.1%}",
                    f"{metrics.get('best_day', 0):.2%}",
                    f"{metrics.get('worst_day', 0):.2%}",
                    f"{metrics.get('skewness', 0):.2f}",
                    f"{metrics.get('kurtosis', 0):.2f}",
                ]
            }

            st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

        with col2:
            # Portfolio composition summary
            st.markdown("**Portfolio Composition**")
            if not position_metrics.empty:
                # Weight column is stored as percentage points (e.g., 10 for 10%), convert to decimal
                cash_weight_pct = position_metrics[position_metrics['Symbol'] == 'CASH']['Weight'].sum() if 'CASH' in position_metrics['Symbol'].values else 0
                securities_weight_pct = 100 - cash_weight_pct
                num_positions = len(position_metrics[position_metrics['Symbol'] != 'CASH'])

                comp_data = {
                    'Metric': ['Number of Positions', 'Securities Weight', 'Cash Weight', 'Total Portfolio Value'],
                    'Value': [f"{num_positions}", f"{securities_weight_pct:.1f}%", f"{cash_weight_pct:.1f}%", f"${total_value:,.0f}"]
                }
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        # Create summary_df for export compatibility
        summary_df = pd.DataFrame(return_data)

        st.markdown("---")

        # Basel III-style Risk Metrics (Simplified)
        st.markdown("### Basel III Risk Framework (Simplified)")

        with st.expander("ℹ️ About Basel III Metrics", expanded=False):
            st.markdown("""
            This section provides simplified Basel III-style risk metrics:

            - **Risk-Weighted Assets (RWA)**: Assets weighted by risk (using beta as proxy)
            - **Capital Requirement**: Minimum capital buffer (simplified as 8% of RWA)
            - **Liquidity Coverage Ratio (LCR)**: Simplified liquidity measure

            *Note: These are illustrative calculations, not regulatory-compliant figures.*
            """)

        col1, col2, col3 = st.columns(3)

        # Simple RWA calculation (using beta as risk weight proxy)
        if not position_metrics.empty:
            position_metrics['RWA'] = position_metrics['Market Value'] * position_metrics['Beta'].clip(lower=0.5)
            total_rwa = position_metrics['RWA'].sum()

            with col1:
                st.metric(
                    "Risk-Weighted Assets",
                    f"${total_rwa:,.0f}",
                    f"RWA/Total: {total_rwa/total_value:.1%}" if total_value > 0 else "N/A"
                )

            capital_requirement = total_rwa * 0.08
            with col2:
                st.metric(
                    "Capital Requirement (8%)",
                    f"${capital_requirement:,.0f}",
                    "Tier 1 Capital Buffer"
                )

            # Simplified LCR (assume all equity is liquid)
            lcr = total_value / capital_requirement if capital_requirement > 0 else 0
            with col3:
                st.metric(
                    "Liquidity Coverage (Simplified)",
                    f"{lcr:.1f}x",
                    "≥1.0 required" if lcr >= 1 else "⚠️ Below threshold"
                )

        st.markdown("---")

        # Export options
        st.markdown("### Export Reports")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Excel export
            excel_data = export_to_excel({
                'Summary': summary_df,
                'Risk Metrics': pd.DataFrame(risk_data),
                'Holdings': position_metrics,
                'Performance Metrics': pd.DataFrame([metrics])
            })

            st.download_button(
                "📊 Download Excel Report",
                excel_data,
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

        with col2:
            # CSV export
            full_report = pd.concat([
                summary_df.assign(Category='Performance'),
                pd.DataFrame(risk_data).assign(Category='Risk')
            ], ignore_index=True)

            csv_data = full_report.to_csv(index=False)

            st.download_button(
                "📄 Download CSV Report",
                csv_data,
                file_name=f"portfolio_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        with col3:
            # PDF export (if reportlab available)
            pdf_data = generate_pdf_report(metrics, st.session_state.holdings, None)

            if pdf_data:
                st.download_button(
                    "📑 Download PDF Report",
                    pdf_data,
                    file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
            else:
                st.info("PDF export requires reportlab. Install with: pip install reportlab")

        st.markdown("---")

        # Transaction history
        st.markdown("### Transaction History")

        if not st.session_state.transactions.empty:
            st.dataframe(
                st.session_state.transactions.sort_values('date', ascending=False),
                use_container_width=True,
                hide_index=True
            )

            tx_csv = st.session_state.transactions.to_csv(index=False)
            st.download_button(
                "📥 Download Transactions",
                tx_csv,
                file_name="transactions_export.csv",
                mime="text/csv"
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8b8fa3; padding: 1rem 0;">
        <small>Portfolio Risk Dashboard v1.0 | Data from Yahoo Finance | For educational purposes only</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
