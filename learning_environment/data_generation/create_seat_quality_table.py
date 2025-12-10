"""
Create a new table for seat quality scores based on Zone + Section combinations.

This script:
1. Analyzes all ticket sales in the database
2. Groups by unique Zone + Section combinations
3. Computes quality metrics for each combination
4. Creates a new 'seat_quality' table with these mappings
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def analyze_zone_section_combinations(db_path: Path) -> pd.DataFrame:
    """
    Analyze all Zone + Section combinations in the database.
    
    Returns DataFrame with columns:
    - zone, section, total_sales, avg_price, median_price, 
    - avg_quality, median_quality, price_percentile, quality_percentile
    """
    conn = sqlite3.connect(str(db_path))
    
    # Query all sales with zone and section
    query = """
        SELECT 
            Zone,
            Section,
            Price,
            CAST(ticket_quality AS REAL) as quality_score,
            Qty
        FROM ticket_sales
        WHERE Zone IS NOT NULL 
            AND Section IS NOT NULL
            AND Price IS NOT NULL
            AND ticket_quality IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        raise ValueError("No sales data found with Zone and Section")
    
    # Group by Zone + Section
    grouped = df.groupby(['Zone', 'Section']).agg({
        'Price': ['count', 'mean', 'median', 'std'],
        'quality_score': ['mean', 'median', 'std'],
        'Qty': 'sum'
    }).reset_index()
    
    # Flatten column names
    grouped.columns = [
        'zone', 'section', 'total_sales', 'avg_price', 'median_price', 'price_std',
        'avg_quality', 'median_quality', 'quality_std', 'total_qty'
    ]
    
    # Compute percentiles across all zone+section combinations
    # Price percentile: higher price = higher percentile
    price_sorted = grouped.sort_values('median_price')
    price_sorted['price_percentile'] = np.arange(len(price_sorted)) / len(price_sorted)
    
    # Quality percentile: higher quality = higher percentile
    quality_sorted = grouped.sort_values('median_quality')
    quality_sorted['quality_percentile'] = np.arange(len(quality_sorted)) / len(quality_sorted)
    
    # Merge percentiles back
    grouped = grouped.merge(
        price_sorted[['zone', 'section', 'price_percentile']],
        on=['zone', 'section'],
        how='left'
    )
    grouped = grouped.merge(
        quality_sorted[['zone', 'section', 'quality_percentile']],
        on=['zone', 'section'],
        how='left'
    )
    
    # Compute composite quality score
    # Could be based on price percentile, quality percentile, or both
    # For now, use a weighted combination
    grouped['composite_quality'] = (
        0.6 * grouped['price_percentile'] + 
        0.4 * grouped['quality_percentile']
    )
    
    # Add quality tier
    def get_quality_tier(composite_quality: float) -> str:
        if composite_quality >= 0.75:
            return 'Premium'
        elif composite_quality >= 0.50:
            return 'High'
        elif composite_quality >= 0.25:
            return 'Medium'
        else:
            return 'Low'
    
    grouped['quality_tier'] = grouped['composite_quality'].apply(get_quality_tier)
    
    # Sort by composite quality
    grouped = grouped.sort_values('composite_quality', ascending=False).reset_index(drop=True)
    
    return grouped


def create_seat_quality_table(conn: sqlite3.Connection) -> None:
    """Create the seat_quality table if it doesn't exist."""
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS seat_quality (
            seat_quality_id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone TEXT NOT NULL,
            section TEXT NOT NULL,
            total_sales INTEGER,
            avg_price REAL,
            median_price REAL,
            price_std REAL,
            avg_quality REAL,
            median_quality REAL,
            quality_std REAL,
            total_qty INTEGER,
            price_percentile REAL,
            quality_percentile REAL,
            composite_quality REAL,
            quality_tier TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(zone, section)
        )
    """)
    
    # Create index for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_zone_section 
        ON seat_quality(zone, section)
    """)
    
    conn.commit()


def insert_seat_quality_data(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert seat quality data into the table."""
    cursor = conn.cursor()
    
    # Determine quality tier
    def get_quality_tier(composite_quality: float) -> str:
        if composite_quality >= 0.75:
            return 'Premium'
        elif composite_quality >= 0.50:
            return 'High'
        elif composite_quality >= 0.25:
            return 'Medium'
        else:
            return 'Low'
    
    inserted = 0
    for _, row in df.iterrows():
        try:
            quality_tier = get_quality_tier(row['composite_quality'])
            
            cursor.execute("""
                INSERT OR REPLACE INTO seat_quality (
                    zone, section, total_sales, avg_price, median_price,
                    price_std, avg_quality, median_quality, quality_std,
                    total_qty, price_percentile, quality_percentile,
                    composite_quality, quality_tier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['zone'],
                row['section'],
                int(row['total_sales']),
                float(row['avg_price']),
                float(row['median_price']),
                float(row['price_std']) if pd.notna(row['price_std']) else None,
                float(row['avg_quality']),
                float(row['median_quality']),
                float(row['quality_std']) if pd.notna(row['quality_std']) else None,
                int(row['total_qty']),
                float(row['price_percentile']),
                float(row['quality_percentile']),
                float(row['composite_quality']),
                quality_tier
            ))
            inserted += 1
        except Exception as e:
            print(f"Error inserting row for {row['zone']}/{row['section']}: {e}")
            continue
    
    conn.commit()
    return inserted


def main():
    """Main function."""
    db_path = Path('learning_environment/data_generation/db.sqlite')
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return 1
    
    print("Analyzing Zone + Section combinations...")
    df = analyze_zone_section_combinations(db_path)
    
    print(f"\nFound {len(df)} unique Zone + Section combinations")
    print("\nTop 10 by composite quality:")
    print(df[['zone', 'section', 'composite_quality', 'median_price', 'quality_tier']].head(10).to_string(index=False))
    
    print("\nBottom 10 by composite quality:")
    print(df[['zone', 'section', 'composite_quality', 'median_price', 'quality_tier']].tail(10).to_string(index=False))
    
    print("\nQuality tier distribution:")
    print(df['quality_tier'].value_counts())
    
    # Create table and insert data
    conn = sqlite3.connect(str(db_path))
    
    print("\nCreating seat_quality table...")
    create_seat_quality_table(conn)
    
    print("Inserting seat quality data...")
    inserted = insert_seat_quality_data(conn, df)
    
    conn.close()
    
    print(f"\n✓ Successfully inserted {inserted} Zone + Section combinations into seat_quality table")
    print(f"✓ Table created at: {db_path}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

