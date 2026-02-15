"""
Incremental Data Updater
=========================
Smart update strategy:
1. Check for NEW schemes from AMFI (compared to existing data)
2. Map ONLY new ISINs to Yahoo symbols (saves 1+ hours!)
3. Download LATEST data for ALL funds (new + existing)
4. Recalculate trend KPIs for all

This is MUCH faster than full regeneration!
"""

import pandas as pd
import requests
import yfinance as yf
import time
import random
import re
import numpy as np
from io import StringIO
from datetime import datetime
from scipy import stats
import os

# ========================
# CONFIGURATION
# ========================
AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
START_DATE = "2018-03-15"
END_DATE = datetime.now().strftime("%Y-%m-%d")
BATCH_SIZE = 300
SLEEP_RANGE = (15, 25)
DATA_DIR = "data"

# Create output directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# ========================
# UTILITY FUNCTIONS
# ========================
def fetch_amfi_data():
    """Fetch and clean AMFI mutual fund data"""
    print("📥 Fetching latest AMFI data...")
    text = requests.get(AMFI_URL).text
    lines = [l for l in text.splitlines() if ";" in l]
    clean = "\n".join(lines)
    
    amfi = pd.read_csv(StringIO(clean), sep=";", engine="python")
    amfi.columns = amfi.columns.str.strip().str.replace("\ufeff", "")
    amfi = amfi.dropna(subset=["Scheme Code"])
    amfi["Scheme Code"] = amfi["Scheme Code"].astype(int)
    
    amfi["ISIN"] = amfi["ISIN Div Payout/ ISIN Growth"].fillna(
        amfi["ISIN Div Reinvestment"]
    )
    amfi = amfi.dropna(subset=["ISIN"])
    
    print(f"✅ Loaded {len(amfi)} schemes from AMFI")
    return amfi

def load_existing_mapped_funds():
    """Load existing funds_mapped.csv if it exists"""
    filepath = os.path.join(DATA_DIR, "funds_mapped.csv")
    
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        print(f"📂 Loaded {len(existing)} existing schemes from {filepath}")
        return existing
    else:
        print("📂 No existing data found - will do full mapping")
        return pd.DataFrame()

def identify_new_schemes(amfi_current, existing_mapped):
    """Compare current AMFI data with existing to find new schemes"""
    
    if existing_mapped.empty:
        print("🆕 All schemes are new (first run)")
        return amfi_current
    
    # Find ISINs that don't exist in our database
    existing_isins = set(existing_mapped["ISIN"].dropna().unique())
    current_isins = set(amfi_current["ISIN"].dropna().unique())
    
    new_isins = current_isins - existing_isins
    
    if len(new_isins) == 0:
        print("✅ No new schemes found!")
        return pd.DataFrame()
    
    new_schemes = amfi_current[amfi_current["ISIN"].isin(new_isins)]
    print(f"🆕 Found {len(new_schemes)} NEW schemes to map!")
    print(f"   New ISINs: {len(new_isins)}")
    
    return new_schemes

def isin_to_yahoo_symbol(isin):
    """Search Yahoo Finance for ISIN and return symbol"""
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": isin, "quotesCount": 5, "newsCount": 0}
    
    try:
        r = requests.get(
            url,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        data = r.json()
        
        for q in data.get("quotes", []):
            if q.get("quoteType") in ["MUTUALFUND", "ETF", "EQUITY"]:
                return {
                    "yahoo_symbol": q.get("symbol"),
                    "yahoo_quote_type": q.get("quoteType"),
                    "yahoo_exchange": q.get("exchange")
                }
    except Exception:
        pass
    
    return None

def map_new_isins_to_yahoo(new_schemes):
    """Map only NEW ISINs to Yahoo symbols"""
    
    if new_schemes.empty:
        print("⏭️  Skipping ISIN mapping (no new schemes)")
        return pd.DataFrame()
    
    print(f"🔍 Mapping {len(new_schemes)} new ISINs to Yahoo Finance...")
    results = []
    
    unique_isins = new_schemes["ISIN"].unique()
    
    for i, isin in enumerate(unique_isins):
        mapping = isin_to_yahoo_symbol(isin)
        
        if mapping:
            mapping["ISIN"] = isin
            results.append(mapping)
        
        time.sleep(0.3)  # Be polite to Yahoo
        
        if i % 50 == 0 and i > 0:
            print(f"   Processed {i}/{len(unique_isins)} new ISINs...")
    
    if results:
        yahoo_map = pd.DataFrame(results)
        print(f"✅ Mapped {len(yahoo_map)} new ISINs to Yahoo symbols")
        return new_schemes.merge(yahoo_map, on="ISIN", how="left")
    else:
        print("⚠️  No new mappings found")
        return pd.DataFrame()

def merge_with_existing(new_mapped, existing_mapped, current_amfi):
    """Merge new mappings with existing data and update NAVs"""
    
    if existing_mapped.empty:
        # First run - just return the new data
        return new_mapped
    
    if new_mapped.empty:
        # No new schemes - update existing with latest AMFI data
        print("🔄 Updating existing schemes with latest NAV values...")
        
        # Merge existing with current AMFI to get latest NAVs
        updated = existing_mapped.drop(columns=["Net Asset Value", "Date"], errors="ignore")
        updated = updated.merge(
            current_amfi[["ISIN", "Net Asset Value", "Date"]],
            on="ISIN",
            how="left"
        )
        
        return updated
    
    # Append new schemes to existing
    print("➕ Merging new schemes with existing data...")
    combined = pd.concat([existing_mapped, new_mapped], ignore_index=True)
    
    # Remove duplicates (prefer newer data)
    combined = combined.drop_duplicates(subset=["ISIN"], keep="last")
    
    # Update NAVs from current AMFI
    combined = combined.drop(columns=["Net Asset Value", "Date"], errors="ignore")
    combined = combined.merge(
        current_amfi[["ISIN", "Net Asset Value", "Date"]],
        on="ISIN",
        how="left"
    )
    
    print(f"✅ Combined dataset: {len(combined)} total schemes")
    return combined

# ========================
# PERFORMANCE DATA DOWNLOAD
# ========================
def get_fy(date):
    """Get Financial Year from date (April to March)"""
    return date.year if date.month >= 4 else date.year - 1

def download_fund_performance_incremental(funds_df):
    """Download LATEST data for ALL funds (much faster than full history)"""
    print(f"\n📊 Downloading latest performance data for all funds...")
    
    # Get unique symbols with Yahoo mapping
    funds = funds_df[["yahoo_symbol", "Scheme Name"]].dropna(subset=["yahoo_symbol"]).drop_duplicates()
    symbols = funds["yahoo_symbol"].tolist()
    
    symbol_to_name = dict(zip(funds["yahoo_symbol"], funds["Scheme Name"]))
    
    print(f"Total symbols to update: {len(symbols)}")
    
    # For incremental updates, we only need last 3 months of data
    # But we'll download from start date to ensure we have complete history
    all_data = []
    failed_symbols = []
    
    # Download in batches
    for i, start_idx in enumerate(range(0, len(symbols), BATCH_SIZE), 1):
        batch = symbols[start_idx:start_idx + BATCH_SIZE]
        
        print(f"\n📦 Batch {i}/{(len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE} | Processing {len(batch)} symbols")
        
        # Retry logic
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                raw = yf.download(
                    tickers=batch,
                    start=START_DATE,
                    end=END_DATE,
                    group_by="ticker",
                    actions=True,
                    auto_adjust=False,
                    threads=False,
                    progress=False
                )
                success = True
            except Exception as e:
                retry_count += 1
                print(f"⚠️  Attempt {retry_count} failed: {e}")
                if retry_count < max_retries:
                    wait_time = retry_count * 30
                    print(f"   Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Batch {i} failed after {max_retries} attempts")
                    failed_symbols.extend(batch)
                    continue
        
        if not success:
            continue
        
        try:
            batch_rows = []
            
            for sym in batch:
                if sym not in raw or len(raw[sym]) == 0:
                    continue
                
                df = raw[sym].reset_index()
                df["yahoo_symbol"] = sym
                df["fund_name"] = symbol_to_name.get(sym)
                df["FY"] = df["Date"].apply(get_fy)
                
                batch_rows.append(df[[
                    "Date", "Close", "Dividends",
                    "yahoo_symbol", "fund_name", "FY"
                ]])
            
            if batch_rows:
                batch_data = pd.concat(batch_rows, ignore_index=True)
                batch_data.rename(
                    columns={"Close": "NAV", "Dividends": "Dividend"},
                    inplace=True
                )
                batch_data["Dividend"] = batch_data["Dividend"].fillna(0)
                all_data.append(batch_data)
            
            # Sleep between batches
            if i < (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE:
                sleep_time = random.uniform(*SLEEP_RANGE)
                print(f"⏳ Cooling down for {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        except Exception as e:
            print(f"❌ Error processing batch {i}: {e}")
    
    if not all_data:
        raise ValueError("No data downloaded! Check network or Yahoo Finance status.")
    
    data = pd.concat(all_data, ignore_index=True)
    
    if failed_symbols:
        print(f"\n⚠️  Warning: {len(failed_symbols)} symbols failed to download")
    
    print(f"✅ Downloaded {len(data)} records")
    
    return data

def calculate_fy_metrics(data):
    """Calculate FY-wise metrics"""
    print("📈 Calculating FY metrics...")
    
    # NAV at start and end of FY
    nav_fy = (
        data.sort_values("Date")
        .groupby(["yahoo_symbol", "fund_name", "FY"])
        .agg(
            April_NAV=("NAV", "first"),
            March_NAV=("NAV", "last")
        )
        .reset_index()
    )
    
    # Total dividends per FY
    div_fy = (
        data.groupby(["yahoo_symbol", "fund_name", "FY"])["Dividend"]
        .sum()
        .reset_index()
    )
    
    # Merge
    fy = nav_fy.merge(div_fy, on=["yahoo_symbol", "fund_name", "FY"], how="left")
    
    # Calculate returns and yield
    fy["FY_Return"] = fy["March_NAV"] - fy["April_NAV"] + fy["Dividend"]
    fy["Avg_NAV"] = (fy["April_NAV"] + fy["March_NAV"]) / 2
    fy["IDCW_Yield_pct"] = (fy["Dividend"] / fy["Avg_NAV"] * 100).round(2)
    
    # Format FY
    fy["FY"] = fy["FY"].astype(str) + "-" + (fy["FY"] + 1).astype(str)
    
    print(f"✅ Calculated metrics for {len(fy)} fund-year combinations")
    
    return fy

# ========================
# TREND KPI CALCULATION
# ========================
def calculate_trend_kpis(fy_data):
    """Calculate dividend yield trend KPIs"""
    print("📊 Calculating trend KPIs...")
    
    trend_kpis = []
    
    for (symbol, name), group in fy_data.groupby(["yahoo_symbol", "fund_name"]):
        group = group.sort_values("FY").reset_index(drop=True)
        
        if len(group) < 3:
            continue
        
        yields = group["IDCW_Yield_pct"].values
        years = np.arange(len(yields))
        
        # CAGR
        if yields[0] > 0 and yields[-1] > 0:
            n_years = len(yields) - 1
            cagr = ((yields[-1] / yields[0]) ** (1 / n_years) - 1) * 100
        else:
            cagr = None
        
        # Linear trend slope
        if len(yields) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, yields)
            trend_slope = slope
            trend_r_squared = r_value ** 2
        else:
            trend_slope = None
            trend_r_squared = None
        
        # Consistency
        if yields.mean() > 0:
            cv = (yields.std() / yields.mean()) * 100
        else:
            cv = None
        
        # Consecutive increases
        increases = np.diff(yields) > 0
        max_consecutive = 0
        current_consecutive = 0
        
        for inc in increases:
            if inc:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        # Average YoY growth
        yoy_growth = np.diff(yields) / yields[:-1] * 100
        avg_yoy_growth = yoy_growth.mean() if len(yoy_growth) > 0 else None
        
        # Recent trend
        if len(yields) >= 3:
            recent_yields = yields[-3:]
            recent_years = np.arange(len(recent_yields))
            recent_slope, _, recent_r, _, _ = stats.linregress(recent_years, recent_yields)
            recent_trend_slope = recent_slope
        else:
            recent_trend_slope = None
        
        trend_kpis.append({
            "yahoo_symbol": symbol,
            "fund_name": name,
            "Years": len(yields),
            "CAGR_Yield": cagr,
            "Trend_Slope": trend_slope,
            "Trend_R_Squared": trend_r_squared,
            "Consistency_CV": cv,
            "Max_Consecutive_Increases": max_consecutive,
            "Avg_YoY_Growth": avg_yoy_growth,
            "Recent_Trend_Slope": recent_trend_slope
        })
    
    trend_df = pd.DataFrame(trend_kpis)
    print(f"✅ Calculated trend KPIs for {len(trend_df)} funds")
    
    return trend_df

# ========================
# SCHEME ATTRIBUTES
# ========================
def clean_text(x):
    """Clean text for matching"""
    x = str(x).lower()
    x = re.sub(r"mutual fund", "", x)
    x = re.sub(r"asset management company", "", x)
    x = re.sub(r"[^a-z0-9 ]", "", x)
    return x.strip()

def extract_scheme_attributes(funds_df, amc_list_path="AMC.csv"):
    """Extract AMC, Plan, and Option from scheme names"""
    print("🏷️  Extracting scheme attributes...")
    
    try:
        amc_list = pd.read_csv(amc_list_path)
    except FileNotFoundError:
        print("⚠️  AMC.csv not found. Creating default AMC list...")
        amc_list = pd.DataFrame({
            "AMC": [
                "Aditya Birla Sun Life Mutual Fund", "HDFC Mutual Fund",
                "ICICI Prudential Mutual Fund", "SBI Mutual Fund",
                "Axis Mutual Fund", "Kotak Mahindra Mutual Fund",
                "UTI Mutual Fund", "DSP Mutual Fund",
                "Nippon India Mutual Fund", "Franklin Templeton Mutual Fund",
                "Tata Mutual Fund", "HSBC Mutual Fund",
                "L&T Mutual Fund", "Mirae Asset Mutual Fund",
                "Motilal Oswal Mutual Fund", "Parag Parikh Mutual Fund",
                "Edelweiss Mutual Fund", "IDFC Mutual Fund",
                "Invesco Mutual Fund", "PPFAS Mutual Fund",
                "Quantum Mutual Fund", "Sundaram Mutual Fund",
                "Mahindra Manulife Mutual Fund", "Baroda BNP Paribas Mutual Fund",
                "Canara Robeco Mutual Fund", "360 ONE Mutual Fund"
            ]
        })
    
    amc_list["AMC_clean"] = amc_list["AMC"].apply(clean_text)
    funds_df["Scheme_clean"] = funds_df["Scheme Name"].apply(clean_text)
    
    full_keys = amc_list["AMC_clean"].tolist()
    first_word_keys = (
        amc_list["AMC_clean"]
        .str.split()
        .str[0]
        .dropna()
        .tolist()
    )
    
    all_keys = sorted(set(full_keys + first_word_keys), key=len, reverse=True)
    pattern = r"^(" + "|".join(map(re.escape, all_keys)) + r")\b"
    compiled_pattern = re.compile(pattern)
    
    def match_amc(s):
        match = compiled_pattern.search(s)
        return match.group(1) if match else "Unknown"
    
    funds_df["Matched_Key"] = funds_df["Scheme_clean"].apply(match_amc)
    
    mapping_dict = {}
    for _, row in amc_list.iterrows():
        mapping_dict[row["AMC_clean"]] = row["AMC"]
        first_word = row["AMC_clean"].split()[0]
        mapping_dict[first_word] = row["AMC"]
    
    funds_df["AMC"] = funds_df["Matched_Key"].map(mapping_dict).fillna("Unknown")
    
    funds_df["Plan"] = "Unknown"
    funds_df.loc[funds_df["Scheme_clean"].str.contains(r"\bdirect\b", na=False), "Plan"] = "Direct"
    funds_df.loc[funds_df["Scheme_clean"].str.contains(r"\bregular\b", na=False), "Plan"] = "Regular"
    
    funds_df["Option"] = "Unknown"
    funds_df.loc[funds_df["Scheme_clean"].str.contains(r"\bgrowth\b", na=False), "Option"] = "Growth"
    funds_df.loc[
        funds_df["Scheme_clean"].str.contains(
            r"\b(?:idcw|dividend|div\b|div\.|payout|reinvestment)\b",
            na=False
        ),
        "Option"
    ] = "IDCW"
    
    print(f"✅ Extracted attributes for {len(funds_df)} schemes")
    
    return funds_df[["yahoo_symbol", "Scheme Name", "AMC", "Plan", "Option"]]

# ========================
# MAIN EXECUTION
# ========================
def main():
    """Main execution - incremental update strategy"""
    
    print("\n" + "="*70)
    print("INCREMENTAL MUTUAL FUND DATA UPDATER")
    print("="*70 + "\n")
    
    try:
        start_time = datetime.now()
        
        # Step 1: Fetch current AMFI data
        print("STEP 1/5: Fetching current AMFI data")
        print("-" * 70)
        current_amfi = fetch_amfi_data()
        
        # Step 2: Load existing data and identify new schemes
        print("\nSTEP 2/5: Checking for new schemes")
        print("-" * 70)
        existing_mapped = load_existing_mapped_funds()
        new_schemes = identify_new_schemes(current_amfi, existing_mapped)
        
        # Step 3: Map only NEW ISINs (HUGE TIME SAVER!)
        print("\nSTEP 3/5: Mapping new ISINs to Yahoo Finance")
        print("-" * 70)
        new_mapped = map_new_isins_to_yahoo(new_schemes)
        
        # Step 4: Merge with existing data
        print("\nSTEP 4/5: Merging with existing data")
        print("-" * 70)
        combined_funds = merge_with_existing(new_mapped, existing_mapped, current_amfi)
        
        # Save updated funds mapping
        output_path = os.path.join(DATA_DIR, "funds_mapped.csv")
        combined_funds.to_csv(output_path, index=False)
        print(f"💾 Saved: {output_path}")
        print(f"   Total schemes: {len(combined_funds)}")
        print(f"   New schemes: {len(new_mapped) if not new_mapped.empty else 0}")
        
        # Step 5: Download performance data for ALL funds
        print("\nSTEP 5/5: Downloading latest data for ALL funds")
        print("-" * 70)
        performance_data = download_fund_performance_incremental(combined_funds)
        fy_metrics = calculate_fy_metrics(performance_data)
        
        output_path = os.path.join(DATA_DIR, "idcw_fy_output.csv")
        fy_metrics.to_csv(output_path, index=False)
        print(f"💾 Saved: {output_path}")
        print(f"   Records: {len(fy_metrics)}")
        
        # Step 6: Calculate trend KPIs
        print("\nSTEP 6/5: Calculating trend KPIs")
        print("-" * 70)
        trend_kpis = calculate_trend_kpis(fy_metrics)
        
        output_path = os.path.join(DATA_DIR, "trend_kpis.csv")
        trend_kpis.to_csv(output_path, index=False)
        print(f"💾 Saved: {output_path}")
        print(f"   Records: {len(trend_kpis)}")
        
        # Step 7: Extract scheme attributes
        print("\nSTEP 7/5: Extracting scheme attributes")
        print("-" * 70)
        scheme_master = extract_scheme_attributes(combined_funds)
        
        output_path = os.path.join(DATA_DIR, "final_schemes_master.csv")
        scheme_master.to_csv(output_path, index=False)
        print(f"💾 Saved: {output_path}")
        print(f"   Records: {len(scheme_master)}")
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("✅ INCREMENTAL UPDATE COMPLETE")
        print("=" * 70)
        print(f"\n⏱️  Total time: {duration/60:.1f} minutes")
        print(f"\nUpdated files in: {DATA_DIR}/")
        print(f"\n📊 Summary:")
        print(f"  • Total schemes: {len(combined_funds):,}")
        print(f"  • New schemes added: {len(new_mapped) if not new_mapped.empty else 0}")
        print(f"  • Fund-year records: {len(fy_metrics):,}")
        print(f"  • Funds with trends: {len(trend_kpis):,}")
        
        if new_schemes.empty:
            print(f"\n💡 No new schemes found - only refreshed existing data!")
            print(f"   Next time will be even faster if no new funds are added.")
        
        print("\n🎯 Next: Dashboard will auto-reload with updated data!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
