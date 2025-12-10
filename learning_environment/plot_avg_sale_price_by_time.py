"""
Plot average sale price percentage vs time from database, normalized for quality.

This helps identify if expensive items near the end of episodes have unrealistic
bias to sell, or if this pattern exists in the actual data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
from pathlib import Path
import sys
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Add parent directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from demand_modeling.data_extractor import load_database


def smooth_data(x, y, method='savgol', window_length=None, polyorder=3):
    """
    Smooth data using Savitzky-Golay filter or interpolation.
    
    Args:
        x: x values
        y: y values
        method: 'savgol' for Savitzky-Golay filter, 'interp' for interpolation
        window_length: Window length for savgol (must be odd, < len(y))
        polyorder: Polynomial order for savgol
    """
    if len(y) < 3:
        return y
    
    if method == 'savgol':
        # Use Savitzky-Golay filter for smoothing
        if window_length is None:
            window_length = min(len(y) // 3 * 2, 51)  # Default: ~2/3 of data, max 51
            if window_length % 2 == 0:
                window_length -= 1  # Must be odd
            window_length = max(3, window_length)  # At least 3
        
        if window_length >= len(y):
            window_length = len(y) - 1 if len(y) % 2 == 0 else len(y) - 2
        
        if window_length < 3:
            return y
        
        try:
            return savgol_filter(y, window_length, polyorder)
        except:
            return y
    else:
        # Interpolation-based smoothing
        f = interp1d(x, y, kind='cubic', fill_value='extrapolate')
        x_smooth = np.linspace(x.min(), x.max(), len(x) * 3)
        y_smooth = f(x_smooth)
        # Downsample back to original x points
        f_back = interp1d(x_smooth, y_smooth, kind='linear', fill_value='extrapolate')
        return f_back(x)


def plot_avg_sale_price_by_time(
    db_path: Path,
    output_path: Path = None,
    max_time_hours: float = 1000.0  # ~42 days = 1000 hours
):
    """
    Plot average sale price (dollars) vs time, normalized for quality only.
    
    Shows actual prices (not percentages) and normalizes only for quality effects,
    without using reference price normalization. This allows seeing absolute price
    trends over time.
    
    Args:
        db_path: Path to SQLite database
        output_path: Optional path to save figure
        max_time_hours: Maximum time_to_event to consider (default 1000h)
    """
    conn = load_database(db_path)
    
    # Load all sales with event context
    query = """
        SELECT 
            ts.event_id,
            ts.time_to_event,
            ts.Price,
            ts.ticket_quality,
            ts.Qty,
            e.day_of_week,
            e.year,
            e.month
        FROM ticket_sales ts
        JOIN events e ON ts.event_id = e.event_id
        WHERE ts.time_to_event IS NOT NULL
            AND ts.time_to_event >= 0
            AND ts.time_to_event <= ?
            AND ts.Price IS NOT NULL
            AND ts.ticket_quality IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn, params=(max_time_hours,))
    
    if len(df) == 0:
        raise ValueError("No sales data found in database")
    
    # Compute quality tiers and scores
    df['quality_score'] = df['ticket_quality'].astype(float)
    df['quality_tier'] = df['quality_score'].apply(
        lambda x: 'Premium' if x >= 0.75 
        else 'High' if x >= 0.50 
        else 'Medium' if x >= 0.25 
        else 'Low'
    )
    
    # Use actual prices (no reference price normalization)
    # We'll normalize for quality effects later
    df['price'] = df['Price'].astype(float)
    
    # Create fine-grained time bins for continuous plotting
    # Use 6-hour bins for smooth curves
    bin_size_hours = 6.0
    
    def compute_time_bin_fine(time_to_event):
        return int(time_to_event // bin_size_hours)
    
    df['time_bin'] = df['time_to_event'].apply(compute_time_bin_fine)
    
    # Time bin centers for plotting (middle of each 6-hour bin)
    # We'll compute these dynamically based on actual bins in data
    
    # Group by time_bin and compute average price (dollars)
    # FIXED: Only normalize by event, NOT by quality (quality has circular dependency with time)
    # Quality includes clearance_score which depends on time_to_event, so normalizing by quality
    # removes the time effect we're trying to measure!
    results = []
    
    # Step 1: Normalize prices by event only (remove event-specific scale effects)
    # This allows us to see true price trends over time without circular dependency
    event_medians = df.groupby('event_id')['price'].median()
    df['price_normalized_by_event'] = df.apply(
        lambda row: row['price'] / event_medians[row['event_id']] if event_medians[row['event_id']] > 0 else row['price'],
        axis=1
    )
    
    # Step 2: For each time bin, compute average event-normalized price
    # We'll use the global median price to convert back to dollar scale
    global_median_price = df['price'].median()
    
    for time_bin in sorted(df['time_bin'].unique()):
        bin_data = df[df['time_bin'] == time_bin]
        
        if len(bin_data) == 0:
            continue
        
        # Time center: middle of the 6-hour bin
        time_center = (time_bin * bin_size_hours) + (bin_size_hours / 2)
        
        # Compute overall average price (dollars), weighted by quantity
        overall_avg_price = np.average(
            bin_data['price'],
            weights=bin_data['Qty']
        )
        
        # Compute average event-normalized price (this removes event scale effects)
        # Convert back to dollar scale using global median
        avg_normalized_price = np.average(
            bin_data['price_normalized_by_event'],
            weights=bin_data['Qty']
        )
        
        # Normalized price: event-normalized price converted back to dollars
        # This shows price trends over time, normalized for event scales only
        # (NOT normalized for quality, to avoid circular dependency)
        normalized_price = avg_normalized_price * global_median_price
        
        # Also compute deviation from global median for comparison
        price_deviation = normalized_price - global_median_price
        
        results.append({
            'time_bin': time_bin,
            'time_center': time_center,
            'avg_price': overall_avg_price,
            'normalized_price': normalized_price,
            'price_deviation': price_deviation,
            'n_sales': len(bin_data),
            'total_qty': bin_data['Qty'].sum()
        })
    
    results_df = pd.DataFrame(results)
    # Sort by time_center to ensure proper plotting order
    results_df = results_df.sort_values('time_center')
    
    # Smooth the normalized price data
    if len(results_df) > 3:
        results_df['normalized_price_smooth'] = smooth_data(
            results_df['time_center'].values,
            results_df['normalized_price'].values,
            method='savgol'
        )
        results_df['price_deviation_smooth'] = smooth_data(
            results_df['time_center'].values,
            results_df['price_deviation'].values,
            method='savgol'
        )
    else:
        results_df['normalized_price_smooth'] = results_df['normalized_price']
        results_df['price_deviation_smooth'] = results_df['price_deviation']
    
    # Create plots: 5 graphs (1000h, 2000h, 6000h, 120h zoom, and profit analysis)
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: 1000 hours
    ax1 = plt.subplot(3, 2, 1)
    # Plot 2: 2000 hours  
    ax2 = plt.subplot(3, 2, 2)
    # Plot 3: 6000 hours
    ax3 = plt.subplot(3, 2, 3)
    # Plot 4: Last 120 hours (zoomed)
    ax4 = plt.subplot(3, 2, 4)
    
    # Helper function to plot normalized price
    def plot_normalized_price(ax, data_df, max_hours, title_suffix=""):
        plot_data = data_df[data_df['time_center'] <= max_hours].copy()
        if len(plot_data) == 0:
            return
        
        # Plot both raw and smoothed
        ax.plot(
            plot_data['time_center'],
            plot_data['normalized_price'],
            marker='o',
            linewidth=1.5,
            markersize=4,
            color='#FF6B6B',
            alpha=0.4,
            label='Raw Data'
        )
        ax.plot(
            plot_data['time_center'],
            plot_data['normalized_price_smooth'],
            linewidth=3,
            color='#FF6B6B',
            label='Smoothed'
        )
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Time to Event (hours)', fontsize=11)
        ax.set_ylabel('Event-Normalized Sale Price ($)', fontsize=11)
        ax.set_title(f'Event-Normalized Sale Price vs Time{title_suffix}\n(0-{int(max_hours)} hours, Event Scale Only)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Set tighter y-axis limits based on smoothed data range
        y_data = plot_data['normalized_price_smooth'].values
        y_min = np.nanmin(y_data)
        y_max = np.nanmax(y_data)
        y_range = y_max - y_min
        # Add 10% padding on each side
        y_padding = y_range * 0.1
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Add time markers
        time_markers = [24, 72, 168, 336, 720, 1000, 2000, 3000, 6000]
        for time_marker in time_markers:
            if time_marker <= max_hours:
                ax.axvline(x=time_marker, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        # Add annotations
        marker_labels = {
            24: '1 day', 72: '3 days', 168: '1 week', 336: '2 weeks',
            720: '30 days', 1000: '42 days', 2000: '83 days', 3000: '125 days', 6000: '250 days'
        }
        y_max_lim = ax.get_ylim()[1]
        for time_marker, label in marker_labels.items():
            if time_marker <= max_hours:
                ax.text(time_marker, y_max_lim * 0.95, label, rotation=90, ha='right', fontsize=7)
    
    # Plot 1: 1000 hours
    plot_normalized_price(ax1, results_df, 1000.0, " (1000h)")
    
    # Plot 2: 2000 hours
    plot_normalized_price(ax2, results_df, 2000.0, " (2000h)")
    
    # Plot 3: 6000 hours
    plot_normalized_price(ax3, results_df, 6000.0, " (6000h)")
    
    # Filter data for last 120 hours and create very fine bins
    df_last120 = df[df['time_to_event'] <= 120].copy()
    
    if len(df_last120) > 0:
        # Create very fine time bins for last 120 hours (every 2 hours for high resolution)
        bin_size_120h = 2.0
        
        def compute_time_bin_120h(time_to_event):
            return int(time_to_event // bin_size_120h)
        
        df_last120['time_bin_120'] = df_last120['time_to_event'].apply(compute_time_bin_120h)
        
        # Compute quality-normalized prices for last 120 hours
        results_last120 = []
        
        # Use same normalization approach (event only, no quality)
        # Ensure event-normalized prices are computed for last120 data
        event_medians_120 = df_last120.groupby('event_id')['price'].median()
        df_last120['price_normalized_by_event'] = df_last120.apply(
            lambda row: row['price'] / event_medians_120[row['event_id']] if event_medians_120[row['event_id']] > 0 else row['price'],
            axis=1
        )
        
        for time_bin_120 in sorted(df_last120['time_bin_120'].unique()):
            bin_data = df_last120[df_last120['time_bin_120'] == time_bin_120]
            
            if len(bin_data) == 0:
                continue
            
            # Time center for this bin (middle of 2-hour window)
            time_center_120 = (time_bin_120 * bin_size_120h) + (bin_size_120h / 2)
            
            overall_avg_price = np.average(
                bin_data['price'],
                weights=bin_data['Qty']
            )
            
            # Compute average event-normalized price
            avg_normalized_price = np.average(
                bin_data['price_normalized_by_event'],
                weights=bin_data['Qty']
            )
            
            # Normalized price: event-normalized price converted back to dollars
            normalized_price = avg_normalized_price * global_median_price
            
            results_last120.append({
                'time_bin': time_bin_120,
                'time_center': time_center_120,
                'avg_price': overall_avg_price,
                'normalized_price': normalized_price,
                'n_sales': len(bin_data),
                'total_qty': bin_data['Qty'].sum()
            })
        
        results_last120_df = pd.DataFrame(results_last120)
        # Sort by time_center to ensure proper plotting order
        results_last120_df = results_last120_df.sort_values('time_center')
        
        # Smooth the 120h data
        if len(results_last120_df) > 3:
            results_last120_df['normalized_price_smooth'] = smooth_data(
                results_last120_df['time_center'].values,
                results_last120_df['normalized_price'].values,
                method='savgol'
            )
        else:
            results_last120_df['normalized_price_smooth'] = results_last120_df['normalized_price']
        
        # Plot 4: Last 120 hours (zoomed)
        ax4.plot(
            results_last120_df['time_center'],
            results_last120_df['normalized_price'],
            marker='o',
            linewidth=1.5,
            markersize=4,
            color='#FF6B6B',
            alpha=0.4,
            label='Raw Data'
        )
        ax4.plot(
            results_last120_df['time_center'],
            results_last120_df['normalized_price_smooth'],
            linewidth=3,
            color='#FF6B6B',
            label='Smoothed'
        )
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Time to Event (hours)', fontsize=11)
        ax4.set_ylabel('Event-Normalized Sale Price ($)', fontsize=11)
        ax4.set_title('Event-Normalized Sale Price vs Time (Last 120 Hours)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
        
        # Set tighter y-axis limits based on smoothed data range
        y_data_120 = results_last120_df['normalized_price_smooth'].values
        y_min_120 = np.nanmin(y_data_120)
        y_max_120 = np.nanmax(y_data_120)
        y_range_120 = y_max_120 - y_min_120
        # Add 10% padding on each side
        y_padding_120 = y_range_120 * 0.1
        ax4.set_ylim(y_min_120 - y_padding_120, y_max_120 + y_padding_120)
        
        # Add time markers
        for time_marker in [24, 72, 120]:
            ax4.axvline(x=time_marker, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        # Add annotations
        y_max_120_lim = ax4.get_ylim()[1]
        ax4.text(24, y_max_120_lim * 0.95, '1 day', rotation=90, ha='right', fontsize=7)
        ax4.text(72, y_max_120_lim * 0.95, '3 days', rotation=90, ha='right', fontsize=7)
        ax4.text(120, y_max_120_lim * 0.95, '5 days', rotation=90, ha='right', fontsize=7)
    
    # Plot 5: Profit Analysis
    ax5 = plt.subplot(3, 2, 5)
    
    # Analyze profit potential: normalized price represents deviation from expected
    profit_data = results_df[results_df['time_center'] <= 2000].copy()
    
    if len(profit_data) > 0:
        # Positive normalized price = opportunity to sell above expected
        profit_data['profit_opportunity'] = profit_data['normalized_price_smooth'].clip(lower=0)
        profit_data['loss_risk'] = profit_data['normalized_price_smooth'].clip(upper=0)
        
        ax5.fill_between(
            profit_data['time_center'],
            0,
            profit_data['profit_opportunity'],
            alpha=0.6,
            color='green',
            label='Profit Opportunity'
        )
        ax5.fill_between(
            profit_data['time_center'],
            profit_data['loss_risk'],
            0,
            alpha=0.6,
            color='red',
            label='Loss Risk'
        )
        ax5.plot(
            profit_data['time_center'],
            profit_data['normalized_price_smooth'],
            linewidth=2,
            color='black',
            label='Net Price Deviation'
        )
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax5.set_xlabel('Time to Event (hours)', fontsize=11)
        ax5.set_ylabel('Price Deviation ($)', fontsize=11)
        ax5.set_title('Profit/Loss Analysis: Price Deviation from Expected\n(Green = Profit Opportunity, Red = Loss Risk)', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=9)
        
        # Set tighter y-axis limits based on data range
        y_data_profit = profit_data['normalized_price_smooth'].values
        y_min_profit = np.nanmin(y_data_profit)
        y_max_profit = np.nanmax(y_data_profit)
        y_range_profit = y_max_profit - y_min_profit
        # Add 10% padding on each side
        y_padding_profit = y_range_profit * 0.1
        ax5.set_ylim(y_min_profit - y_padding_profit, y_max_profit + y_padding_profit)
        
        # Add summary statistics
        avg_profit = profit_data['normalized_price_smooth'].mean()
        std_profit = profit_data['normalized_price_smooth'].std()
        positive_pct = (profit_data['normalized_price_smooth'] > 0).sum() / len(profit_data) * 100
        
        summary_text = f"Avg Deviation: ${avg_profit:.2f}\nStd Dev: ${std_profit:.2f}\nPositive: {positive_pct:.1f}%"
        ax5.text(0.02, 0.98, summary_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 6: Agent Profitability Analysis
    ax6 = plt.subplot(3, 2, 6)
    
    # Analyze if agent can find profitable strategies
    if len(profit_data) > 0:
        # Find local minima (buy opportunities) and maxima (sell opportunities)
        from scipy.signal import find_peaks
        
        smooth_prices = profit_data['normalized_price_smooth'].values
        times = profit_data['time_center'].values
        
        # Find peaks (sell opportunities) and troughs (buy opportunities)
        peaks, _ = find_peaks(smooth_prices, height=10, distance=10)
        troughs, _ = find_peaks(-smooth_prices, height=10, distance=10)
        
        ax6.plot(times, smooth_prices, linewidth=2, color='blue', label='Price Deviation', alpha=0.7)
        ax6.scatter(times[peaks], smooth_prices[peaks], color='green', s=100, marker='^', 
                   label='Sell Opportunities', zorder=5)
        ax6.scatter(times[troughs], smooth_prices[troughs], color='red', s=100, marker='v',
                   label='Buy Opportunities', zorder=5)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax6.set_xlabel('Time to Event (hours)', fontsize=11)
        ax6.set_ylabel('Price Deviation ($)', fontsize=11)
        ax6.set_title('Agent Strategy Analysis: Buy Low, Sell High\n(Can agent exploit price patterns?)', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=9)
        
        # Set tighter y-axis limits based on data range (including scatter points)
        y_data_strategy = smooth_prices
        if len(peaks) > 0:
            y_data_strategy = np.concatenate([y_data_strategy, smooth_prices[peaks]])
        if len(troughs) > 0:
            y_data_strategy = np.concatenate([y_data_strategy, smooth_prices[troughs]])
        y_min_strategy = np.nanmin(y_data_strategy)
        y_max_strategy = np.nanmax(y_data_strategy)
        y_range_strategy = y_max_strategy - y_min_strategy
        # Add 10% padding on each side
        y_padding_strategy = y_range_strategy * 0.1
        ax6.set_ylim(y_min_strategy - y_padding_strategy, y_max_strategy + y_padding_strategy)
        
        # Calculate potential profit from buy-low-sell-high strategy
        if len(peaks) > 0 and len(troughs) > 0:
            # Simple strategy: buy at troughs, sell at next peak
            potential_profits = []
            for trough_idx in troughs:
                # Find next peak after this trough
                next_peaks = peaks[peaks > trough_idx]
                if len(next_peaks) > 0:
                    next_peak = next_peaks[0]
                    profit = smooth_prices[next_peak] - smooth_prices[trough_idx]
                    potential_profits.append(profit)
            
            if len(potential_profits) > 0:
                avg_profit_per_trade = np.mean(potential_profits)
                max_profit_per_trade = np.max(potential_profits)
                total_potential = np.sum([p for p in potential_profits if p > 0])
                
                strategy_text = f"Buy-Low-Sell-High Strategy:\n"
                strategy_text += f"Avg Profit/Trade: ${avg_profit_per_trade:.2f}\n"
                strategy_text += f"Max Profit/Trade: ${max_profit_per_trade:.2f}\n"
                strategy_text += f"Total Potential: ${total_potential:.2f}\n"
                strategy_text += f"Profitable Trades: {sum(1 for p in potential_profits if p > 0)}/{len(potential_profits)}"
                
                ax6.text(0.02, 0.02, strategy_text, transform=ax6.transAxes,
                        fontsize=9, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Add text box with summary statistics
    summary_text = f"Total Sales: {df['Qty'].sum():,}\n"
    summary_text += f"Time Range: 0-{max_time_hours}h\n"
    summary_text += f"Price Range: ${df['price'].min():.2f} - ${df['price'].max():.2f}\n"
    summary_text += f"Average Price: ${df['price'].mean():.2f}\n"
    summary_text += f"\nNormalized for event scale only\n(quality normalization removed due to circular dependency)"
    
    fig.text(0.02, 0.02, summary_text, fontsize=9, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        print("\n" + "="*80)
        print("PROFIT ANALYSIS SUMMARY")
        print("="*80)
        profit_data = results_df[results_df['time_center'] <= 2000].copy()
        if len(profit_data) > 0:
            avg_dev = profit_data['normalized_price_smooth'].mean()
            std_dev = profit_data['normalized_price_smooth'].std()
            positive_pct = (profit_data['normalized_price_smooth'] > 0).sum() / len(profit_data) * 100
            
            print(f"\nAverage Price Deviation: ${avg_dev:.2f}")
            print(f"Standard Deviation: ${std_dev:.2f}")
            print(f"Percentage of Time Above Expected: {positive_pct:.1f}%")
            print(f"\nInterpretation:")
            if avg_dev > 0:
                print(f"  ✓ Prices are on average ${avg_dev:.2f} ABOVE expected")
                print(f"  → Resellers can potentially profit by buying early and selling later")
            else:
                print(f"  ✗ Prices are on average ${abs(avg_dev):.2f} BELOW expected")
                print(f"  → Market is competitive; profit requires strategic timing")
            
            if positive_pct > 50:
                print(f"\n  → Agent CAN find profitable strategies ({positive_pct:.1f}% of time above expected)")
                print(f"  → Key: Buy when normalized price < 0, sell when normalized price > 0")
            else:
                print(f"\n  → Agent will struggle ({100-positive_pct:.1f}% of time below expected)")
                print(f"  → Requires very precise timing and demand prediction")
        print("="*80)
    else:
        plt.show()
    
    conn.close()
    
    return results_df


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot average sale price % vs time from database")
    parser.add_argument(
        '--db-path',
        type=Path,
        default=Path(__file__).parent.parent / 'data_generation' / 'db.sqlite',
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--output-path',
        type=Path,
        default=Path(__file__).parent / 'plots' / 'avg_sale_price_by_time.png',
        help='Path to save output plot'
    )
    parser.add_argument(
        '--max-time',
        type=float,
        default=1000.0,
        help='Maximum time_to_event to consider (hours, default 1000 = ~42 days)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plot interactively (in addition to saving)'
    )
    
    args = parser.parse_args()
    
    # Check if database exists
    if not args.db_path.exists():
        print(f"Error: Database file not found: {args.db_path}")
        return 1
    
    # Create output directory
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading sales data from database...")
    results_df = plot_avg_sale_price_by_time(
        db_path=args.db_path,
        output_path=args.output_path,
        max_time_hours=args.max_time
    )
    
    print(f"\nSummary by time bin:")
    print(results_df[['time_bin', 'time_center', 'avg_price', 'normalized_price', 'n_sales']].to_string())
    
    if args.show:
        plt.show()
    else:
        plt.close()
    
    print("\nPlot generated successfully!")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

