# Data Generation

This folder contains scripts and utilities for processing raw ticket sales data and preparing it for model training.

## Contents

- **`import_data.py`**: Main script to import CSV ticket sales data into SQLite database. Processes raw CSV files from `data/TD_Garden/` and stores structured data in `db.sqlite`.

- **`create_seat_quality_table.py`**: Creates the seat quality table in the database schema.

- **`import_utils/`**: Utility modules for data import
  - **`csv_processor.py`**: Processes CSV files and extracts ticket sales records
  - **`filename_parser.py`**: Parses CSV filenames to extract event metadata
  - **`ticket_quality.py`**: Handles seat quality scoring and tier assignment

- **`db/`**: Database schema and utilities
  - **`schema.py`**: Defines SQLite database schema (events, ticket_sales, seat_quality tables)

- **`data/TD_Garden/`**: Raw CSV files containing historical ticket sales data (50 Boston Celtics games)

- **`db.sqlite`**: SQLite database containing processed ticket sales data

- **`seat_quality.csv`**: CSV file with seat quality scores (generated using AI-based quality assessment)

## Usage

To import data:
```bash
cd learning_environment/data_generation
python import_data.py
```

The database (`db.sqlite`) is used by the demand modeling pipeline to extract training data.

