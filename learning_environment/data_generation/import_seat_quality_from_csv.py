"""
Import seat_quality table from CSV, replacing existing data.

This script reads the CSV file and populates/updates the seat_quality table.
"""

import sqlite3
import csv
from pathlib import Path
from typing import List, Dict


def clear_seat_quality_table(conn: sqlite3.Connection) -> None:
    """Clear all existing data from seat_quality table."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM seat_quality")
    conn.commit()
    print("Cleared existing seat_quality data")


def import_csv_to_seat_quality(csv_path: Path, db_path: Path) -> int:
    """
    Import CSV data into seat_quality table.
    
    Returns number of rows inserted.
    """
    conn = sqlite3.connect(str(db_path))
    
    # Read CSV to see what columns we have
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            print("CSV file is empty")
            return 0
        
        # Get column names
        columns = reader.fieldnames
        print(f"CSV columns: {columns}")
        
        # Clear existing data
        clear_seat_quality_table(conn)
        
        cursor = conn.cursor()
        inserted = 0
        
        for row in rows:
            try:
                # Extract zone and section (required)
                zone = row.get('zone', '').strip()
                section = row.get('section', '').strip()
                
                if not zone or not section:
                    print(f"Skipping row with missing zone/section: {row}")
                    continue
                
                # Extract quality_score and rationale (optional)
                quality_score_str = row.get('quality_score', '').strip()
                quality_score = float(quality_score_str) if quality_score_str else None
                rationale = row.get('rationale', '').strip() or None
                
                # Insert or replace based on zone + section
                cursor.execute("""
                    INSERT INTO seat_quality (
                        zone, section, quality_score, rationale
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(zone, section) DO UPDATE SET
                        zone = excluded.zone,
                        section = excluded.section,
                        quality_score = excluded.quality_score,
                        rationale = excluded.rationale
                """, (zone, section, quality_score, rationale))
                
                inserted += 1
            except Exception as e:
                print(f"Error inserting row {row}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        return inserted


def main():
    csv_path = Path('learning_environment/data_generation/seat_quality.csv')
    db_path = Path('learning_environment/data_generation/db.sqlite')
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return 1
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return 1
    
    print(f"Importing {csv_path} into seat_quality table...")
    count = import_csv_to_seat_quality(csv_path, db_path)
    
    print(f"\n✓ Successfully imported {count} rows into seat_quality table")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

