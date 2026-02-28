"""
Data Loader Script
Downloads historical financial data from Yahoo Finance
Author: Murengezi Kevin
Date: 2026
"""

import yfinance as yf
import pandas_datareader as pdr
from pathlib import Path
from datetime import datetime
import warnings

# Suppress pandas warnings
warnings.filterwarnings("ignore")


def setup_directories():
    """
    Create the data/raw directory structure if it doesn't exist.
    Uses pathlib for cross-platform compatibility.
    """
    # Get the script's directory and navigate to project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Define the raw data directory path
    raw_data_dir = project_root / "data" / "raw"
    # Create directory if it doesn't exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"data directory ready: {raw_data_dir}")
    return raw_data_dir


def download_yahoo_data(ticker, output_filename, data_dir, start_date, end_date):
    """
    Download historical data from Yahoo Finance and save to CSV.

    Args:
        ticker (str): Yahoo Finance ticker symbol
        output_filename (str): Output CSV filename
        data_dir (Path): Directory where to save the file
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        print(f"Downloading {ticker} from Yahoo Finance...")
        # Download data using yfinance
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # Check if data is empty
        if data.empty:
            print(f"Error: No data retrieved for {ticker}")
            return False

        # Save to CSV
        output_path = data_dir / output_filename
        data.to_csv(output_path)
        print(f"Data for {ticker} saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return False


def main():
    """Main function to set up directories and download data."""

    print("=" * 70)
    print("RISK DASHBOARD - DATA LOADER")
    print("=" * 70)

    # Setup directories
    data_dir = setup_directories()

    # Define date range
    start_date = "2005-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"\n📅 Date Range: {start_date} to {end_date}")
    print(f"📊 Frequency: Daily")
    print("\n" + "=" * 70)

    # Track download statistics
    total_downloads = 0
    successful_downloads = 0

    # ========================================================================
    # YAHOO FINANCE DOWNLOADS
    # ========================================================================

    yahoo_datasets = [
        {
            "ticker": "BTC-USD",
            "filename": "bitcoin_price.csv",
            "description": "Bitcoin Price",
        },
        {
            "ticker": "USDCHF=X",
            "filename": "usd_chf.csv",
            "description": "USD/CHF Exchange Rate",
        },
        {
            "ticker": "EURCHF=X",
            "filename": "eur_chf.csv",
            "description": "EUR/CHF Exchange Rate",
        },
        {
            "ticker": "^SSMI",
            "filename": "swiss_market_index.csv",
            "description": "Swiss Market Index",
        },
    ]

    print("\n📈 YAHOO FINANCE DOWNLOADS")
    print("-" * 70)
    for dataset in yahoo_datasets:
        total_downloads += 1
        if download_yahoo_data(
            dataset["ticker"], dataset["filename"], data_dir, start_date, end_date
        ):
            successful_downloads += 1
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"✓ Successful: {successful_downloads}/{total_downloads}")
    print(f"✗ Failed: {total_downloads - successful_downloads}/{total_downloads}")

    if successful_downloads == total_downloads:
        print("\n🎉 All data files are ready!")
    else:
        print(
            f"\n⚠️  Warning: {total_downloads - successful_downloads} file(s) missing or failed"
        )

    print(f"\n✅ Data loading process completed!")
    print(f"📁 All files location: {data_dir}\n")


if __name__ == "__main__":
    main()
