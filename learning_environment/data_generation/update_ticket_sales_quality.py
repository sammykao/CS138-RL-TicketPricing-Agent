"""
Update all ticket_sales records to use quality scores from seat_quality table.

This replaces the old computed ticket_quality with the new seat-based quality scores.
"""

import sqlite3
from pathlib import Path
from typing import Optional


def update_ticket_sales_quality(db_path: Path, use_row_adjustment: bool = False) -> dict:
    """
    Update ticket_quality in ticket_sales table using seat_quality table.
    
    Args:
        db_path: Path to database
        use_row_adjustment: If True, adjust quality based on Row (front 5 rows get +0.05)
    
    Returns:
        Dictionary with update statistics
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get all sales with Zone and Section
    cursor.execute("""
        SELECT sale_id, Zone, Section, Row
        FROM ticket_sales
        WHERE Zone IS NOT NULL 
            AND Section IS NOT NULL
    """)
    
    sales = cursor.fetchall()
    print(f"Found {len(sales)} sales with Zone and Section")
    
    # Get quality scores from seat_quality table
    cursor.execute("""
        SELECT zone, section, quality_score
        FROM seat_quality
    """)
    
    quality_map = {}
    for row in cursor.fetchall():
        key = (row[0], row[1])
        quality_map[key] = row[2]
    
    print(f"Loaded {len(quality_map)} quality scores from seat_quality table")
    
    # Update ticket_sales
    updated = 0
    not_found = 0
    errors = 0
    
    front_rows = ['A', 'B', 'C', 'D', 'E', '1', '2', '3', '4', '5']
    
    for sale_id, zone, section, row in sales:
        try:
            key = (zone, section)
            
            if key in quality_map:
                base_quality = quality_map[key]
                
                # Optionally adjust for row
                if use_row_adjustment and row:
                    row_str = str(row).strip().upper()
                    if row_str in front_rows:
                        adjusted_quality = min(1.0, base_quality + 0.05)
                    else:
                        adjusted_quality = base_quality
                else:
                    adjusted_quality = base_quality
                
                # Update ticket_quality
                cursor.execute("""
                    UPDATE ticket_sales
                    SET ticket_quality = ?
                    WHERE sale_id = ?
                """, (f"{adjusted_quality:.4f}", sale_id))
                
                updated += 1
            else:
                not_found += 1
                # Keep existing quality or set to None
                # For now, we'll leave it as is
                
        except Exception as e:
            errors += 1
            if errors <= 5:  # Only print first 5 errors
                print(f"Error updating sale_id {sale_id}: {e}")
    
    conn.commit()
    
    stats = {
        'total_sales': len(sales),
        'updated': updated,
        'not_found': not_found,
        'errors': errors
    }
    
    conn.close()
    return stats


def main():
    db_path = Path('learning_environment/data_generation/db.sqlite')
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return 1
    
    print("Updating ticket_sales with new seat quality scores...")
    print("=" * 80)
    
    # Update without row adjustment first (simpler)
    stats = update_ticket_sales_quality(db_path, use_row_adjustment=False)
    
    print("\n" + "=" * 80)
    print("UPDATE STATISTICS")
    print("=" * 80)
    print(f"Total sales with Zone+Section: {stats['total_sales']}")
    print(f"Successfully updated: {stats['updated']}")
    print(f"Zone+Section not found in seat_quality: {stats['not_found']}")
    print(f"Errors: {stats['errors']}")
    
    if stats['not_found'] > 0:
        print(f"\nNote: {stats['not_found']} sales had Zone+Section combinations")
        print("      not found in seat_quality table. These were not updated.")
    
    print("\n✓ Ticket quality scores updated successfully!")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

