import numpy as np
import pandas as pd
from typing import Dict, Tuple
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.zigzag_indicator import ZigZagIndicator

class SimpleBacktest:
    """
    Simple backtesting framework for ZigZag trading signals.
    """
    
    def __init__(self, initial_capital: float = 10000, 
                 position_size_pct: float = 0.1,
                 slippage: float = 0.001):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            position_size_pct: Percent of capital per trade
            slippage: Slippage as percentage
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.slippage = slippage
    
    def run_backtest(self, df: pd.DataFrame, model_predictions: np.ndarray) -> Dict:
        """
        Run backtest on predictions.
        
        Args:
            df: DataFrame with OHLCV data
            model_predictions: Model predictions (0-4)
            
        Returns:
            Backtest results dictionary
        """
        df = df.copy()
        df['model_signal'] = model_predictions
        
        capital = self.initial_capital
        position = None
        entry_price = None
        trades = []
        equity_curve = [capital]
        
        for i in range(len(df)):
            signal = df.iloc[i]['model_signal']
            close_price = df.iloc[i]['close']
            
            # Close existing position if signal changes
            if position is not None:
                if (position == 'long' and signal in [1, 2]) or \
                   (position == 'short' and signal in [3, 4]):
                    
                    # Close trade
                    if position == 'long':
                        exit_price = close_price * (1 - self.slippage)
                        pnl = (exit_price - entry_price) * (capital * self.position_size_pct / entry_price)
                    else:
                        exit_price = close_price * (1 + self.slippage)
                        pnl = (entry_price - exit_price) * (capital * self.position_size_pct / entry_price)
                    
                    capital += pnl
                    trades.append({
                        'entry_bar': len(trades),
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'pnl': pnl,
                        'pnl_pct': pnl / (capital - pnl) if (capital - pnl) != 0 else 0
                    })
                    position = None
            
            # Open new position
            if position is None:
                if signal in [3, 4]:  # HL, LL = Long
                    position = 'long'
                    entry_price = close_price * (1 + self.slippage)
                elif signal in [1, 2]:  # HH, LH = Short
                    position = 'short'
                    entry_price = close_price * (1 - self.slippage)
            
            equity_curve.append(capital)
        
        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(np.sum([t['pnl'] for t in winning_trades]) / 
                              np.sum([t['pnl'] for t in losing_trades])) if losing_trades else np.inf
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)  # 24/7 for crypto
        max_drawdown = self._calculate_max_drawdown(equity_array)
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'trades': trades,
            'equity_curve': equity_curve
        }
        
        return results
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        """
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown)
        return max_drawdown
    
    def print_results(self, results: Dict):
        """
        Print backtest results in readable format.
        """
        print('\n' + '='*50)
        print('BACKTEST RESULTS')
        print('='*50)
        print(f'Initial Capital: ${results["initial_capital"]:.2f}')
        print(f'Final Capital: ${results["final_capital"]:.2f}')
        print(f'Total Return: {results["total_return_pct"]:.2f}%')
        print(f'\nTrading Statistics:')
        print(f'  Total Trades: {results["total_trades"]}')
        print(f'  Winning: {results["winning_trades"]}')
        print(f'  Losing: {results["losing_trades"]}')
        print(f'  Win Rate: {results["win_rate"]:.2f}%')
        print(f'\nRisk Metrics:')
        print(f'  Sharpe Ratio: {results["sharpe_ratio"]:.4f}')
        print(f'  Max Drawdown: {results["max_drawdown"]:.2f}%')
        print(f'  Profit Factor: {results["profit_factor"]:.4f}')
        print('='*50)
