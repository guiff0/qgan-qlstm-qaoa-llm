import os
import glob
import urllib.request
import datetime
import pandas as pd
import pandas_datareader as pdr
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_EUR_USD
from histdata import download_hist_data
from histdata.api import Platform, TimeFrame

# -------------------------------------------------------------------
# Configuration & Directory Setup
# -------------------------------------------------------------------
BASE_DIR = r"D:\app\qgan-llm-research"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CACHE_DIR = os.path.join(RAW_DIR, "dukascopy_cache")
FOREXSB_DIR = os.path.join(RAW_DIR, "forexsb_chunks")
MASTER_CSV = os.path.join(RAW_DIR, "dukascopy_eurusd_1min_2010_2025.csv")
FRED_CSV = os.path.join(RAW_DIR, "fred_macro_2010_2025.csv")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FOREXSB_DIR, exist_ok=True)

SYMBOL = "EURUSD"
START_YEAR = 2010
CURRENT_YEAR = datetime.datetime.now().year


# -------------------------------------------------------------------
# [1/4] Dukascopy Ingestion (Year-by-Year Cache)
# -------------------------------------------------------------------
def fetch_and_cache_dukascopy_year(year: int):
    """Fetches EURUSD M1 data for a given year and saves it to local cache."""
    cache_file = os.path.join(CACHE_DIR, f"eurusd_1min_{year}.csv")
    
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        print(f" -> [CACHED] Year {year} already cached at {cache_file}. Skipping fetch.")
        return

    now = datetime.datetime.now(datetime.timezone.utc)
    start_date = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
    
    if start_date > now:
        print(f" -> Skipping year {year} (in the future).")
        return

    # Cap end_date to current time for active year
    end_date = now if year == now.year else datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)

    print(f" -> Fetching {SYMBOL} for year {year} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")
    
    try:
        df = dukascopy_python.fetch(
            instrument=INSTRUMENT_FX_MAJORS_EUR_USD,
            interval=dukascopy_python.INTERVAL_MIN_1,
            offer_side=dukascopy_python.OFFER_SIDE_BID,
            start=start_date,
            end=end_date
        )

        if df is None or df.empty:
            print(f"    [WARNING] No data returned for year {year}.")
            return

        df.columns = [str(c).lower().strip() for c in df.columns]
        df.to_csv(cache_file)
        print(f"    [SUCCESS] {year}: {len(df):,} rows cached -> {cache_file}")

    except Exception as e:
        print(f"    [ERROR] Failed to download year {year}: {e}")

def acquire_dukascopy():
    print(f"\n=== [1/4] Downloading Dukascopy {SYMBOL} ({START_YEAR}–{CURRENT_YEAR}) ===")
    for year in range(START_YEAR, CURRENT_YEAR + 1):
        fetch_and_cache_dukascopy_year(year)


# -------------------------------------------------------------------
# [2/4] ForexSB Downloader & Parser
# -------------------------------------------------------------------
def download_forexsb_data(symbol="EURUSD", period="1"):
    """Downloads historical ForexSB CSV data exports directly with fallback mirrors."""
    file_name = f"{symbol}{period}.csv"
    destination_path = os.path.join(FOREXSB_DIR, file_name)

    if os.path.exists(destination_path) and os.path.getsize(destination_path) > 0:
        print(f" -> ForexSB file already exists: {destination_path}")
        return destination_path

    # Secondary mirrors / GitHub releases hosting ForexSB EURUSD1 chunks
    urls = [
        f"https://raw.githubusercontent.com/ForexSB/Forex-Data/master/{file_name}",
        f"https://data.forexsb.com/files/{file_name}"
    ]

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    for url in urls:
        print(f" -> Attempting download from {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response, open(destination_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"    [SUCCESS] Downloaded ForexSB chunk -> {destination_path}")
            return destination_path
        except Exception as e:
            print(f"    [WARNING] Download from {url} failed: {e}")

    print(f" -> [INFO] Could not auto-download ForexSB chunk. Pipeline will continue using Dukascopy data.")
    return None
# -------------------------------------------------------------------
# [2/4] Downloading Historical 1-Minute FX  Data Ingestion
# -------------------------------------------------------------------

BASE_DIR = r"D:\app\qgan-llm-research"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
FOREXSB_DIR = os.path.join(RAW_DIR, "forexsb_chunks")
TARGET_FILE = os.path.join(RAW_DIR, "forexsb_eurusd_1min_2010_2023.csv")

os.makedirs(FOREXSB_DIR, exist_ok=True)

# Public repository hosting full historical ForexSB/MetaTrader M1 exports
FX_DATA_URL = "https://raw.githubusercontent.com/philipperemy/fx-1-minute-data/master/data/eurusd/2010.zip"



def fetch_and_build_forexsb():
    if os.path.exists(TARGET_FILE) and os.path.getsize(TARGET_FILE) > 0:
        print(f" -> Target file already exists: {TARGET_FILE}")
        return

    print("===\n=== [2/4] Downloading Historical 1-Minute FX Data ===")
    
    # Download sample year data using histdata API
    zip_path = download_hist_data(
        year='2010',
        month=None,
        pair='eurusd',
        platform=Platform.GENERIC_ASCII,
        time_frame=TimeFrame.ONE_MINUTE,
        output_directory=FOREXSB_DIR
    )
    
    print(f" -> Download completed: {zip_path}")

    # Unpack and combine extracted CSV files
    all_csvs = glob.glob(os.path.join(FOREXSB_DIR, "*.csv")) + glob.glob(os.path.join(FOREXSB_DIR, "*.txt"))
    if not all_csvs:
        # Check zip files inside output directory if extraction was omitted
        import zipfile
        for zip_file in glob.glob(os.path.join(FOREXSB_DIR, "*.zip")):
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(FOREXSB_DIR)
        all_csvs = glob.glob(os.path.join(FOREXSB_DIR, "*.csv")) + glob.glob(os.path.join(FOREXSB_DIR, "*.txt"))

    if all_csvs:
        # Standardize HistData ASCII schema (DateTime, Open, High, Low, Close, Volume)
        df = pd.read_csv(
            all_csvs[0], 
            sep=';', 
            names=['timestamp', 'open', 'high', 'low', 'close', 'volume'], 
            header=None
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S')
        df.to_csv(TARGET_FILE, index=False)
        print(f" [SUCCESS] Successfully generated {TARGET_FILE} ({len(df):,} rows)")
    else:
        print(" [ERROR] Failed to locate extracted CSV data.")



# -------------------------------------------------------------------
# [3/4] FRED Macroeconomic Data Ingestion
# -------------------------------------------------------------------
def acquire_fred():
    print("\n=== [3/4] Pulling FRED Macroeconomic Data ===")
    
    if os.path.exists(FRED_CSV) and os.path.getsize(FRED_CSV) > 0:
        print(f" -> FRED macro dataset already exists: {FRED_CSV}")
        return

    series_ids = ["CPIAUCSL", "FEDFUNDS", "GS10", "UNRATE", "GDP"]
    try:
        df_fred = pdr.DataReader(series_ids, 'fred', start=f"{START_YEAR}-01-01", end=f"{CURRENT_YEAR}-12-31")
        df_fred.reset_index(inplace=True)
        df_fred.rename(columns={"DATE": "date"}, inplace=True)
        df_fred.to_csv(FRED_CSV, index=False)
        print(f"    [SUCCESS] Pulled {len(df_fred):,} rows from FRED. Saved to: {FRED_CSV}")
    except Exception as e:
        print(f"    [ERROR] Failed to fetch FRED data: {e}")


# -------------------------------------------------------------------
# Dataset Consolidator & Deduplicator
# -------------------------------------------------------------------



# -------------------------------------------------------------------
# [4/4] Pipeline Verification
# -------------------------------------------------------------------
def verify_pipeline():
    print("\n=== [4/4] Pipeline Verification Summary ===")
    expected_files = [
        (os.path.basename(MASTER_CSV), MASTER_CSV),
        (os.path.basename(FRED_CSV), FRED_CSV)
    ]
    
    all_valid = True
    for fname, fpath in expected_files:
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            df = pd.read_csv(fpath, nrows=5)
            print(f" [PASS] {fname:<45} ({size_mb:.2f} MB, ~{len(pd.read_csv(fpath)):,} rows)")
        else:
            print(f" [FAIL] {fname:<45} (Missing or empty)")
            all_valid = False

    if all_valid:
        print("\n -> All data ingestion steps completed successfully!")


# -------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    acquire_dukascopy()
    fetch_and_build_forexsb()
    acquire_fred()
    verify_pipeline()
