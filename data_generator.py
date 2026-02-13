"""
Mutual Fund Data Generator
===========================
This script fetches AMFI data, maps ISINs to Yahoo symbols, downloads historical
performance data, calculates trend KPIs, and saves everything to CSV files in the
data/ folder. No visualization - pure data generation.

Output Files (saved to data/):
- funds_mapped.csv
- idcw_fy_output.csv
- trend_kpis.csv
- final_schemes_master.csv
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
OUTPUT_DIR = "data"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# STEP 1: ISIN TO YAHOO SYMBOL MAPPING
# ========================
def fetch_amfi_data():
    """Fetch and clean AMFI mutual fund data"""
    print("📥 Fetching AMFI data...")
    text = requests.get(AMFI_URL).text
    lines = [l for l in text.splitlines() if ";" in l]
    clean = "\n".join(lines)
    
    amfi = pd.read_csv(StringIO(clean), sep=";", engine="python")
    amfi.columns = amfi.columns.str.strip().str.replace("\ufeff", "")
    amfi = amfi.dropna(subset=["Scheme Code"])
    amfi["Scheme Code"] = amfi["Scheme Code"].astype(int)
    
    # Prefer Growth ISIN when available
    amfi["ISIN"] = amfi["ISIN Div Payout/ ISIN Growth"].fillna(
        amfi["ISIN Div Reinvestment"]
    )
    amfi = amfi.dropna(subset=["ISIN"])
    
    print(f"✅ Loaded {len(amfi)} schemes from AMFI")
    return amfi

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

def map_isins_to_yahoo(amfi):
    """Map all unique ISINs to Yahoo symbols"""
    print("🔍 Mapping ISINs to Yahoo Finance symbols...")
    results = []
    
    for i, isin in enumerate(amfi["ISIN"].unique()):
        mapping = isin_to_yahoo_symbol(isin)
        
        if mapping:
            mapping["ISIN"] = isin
            results.append(mapping)
        
        time.sleep(0.3)  # Be polite to Yahoo
        
        if i % 100 == 0:
            print(f"   Processed {i} ISINs...")
    
    yahoo_map = pd.DataFrame(results)
    print(f"✅ Found Yahoo symbols for {len(yahoo_map)} ISINs")
    
    return amfi.merge(yahoo_map, on="ISIN", how="left")

# ========================
# STEP 2: DOWNLOAD PERFORMANCE DATA
# ========================
def get_fy(date):
    """Get Financial Year from date (April to March)"""
    return date.year if date.month >= 4 else date.year - 1

def download_fund_performance(funds_df):
    """Download NAV and dividend data from Yahoo Finance"""
    print(f"\n📊 Downloading performance data...")
    
    # Remove duplicates and get unique symbols
    funds = funds_df[["yahoo_symbol", "Scheme Name"]].drop_duplicates()
    symbols = funds["yahoo_symbol"].dropna().tolist()
    
    symbol_to_name = dict(zip(funds["yahoo_symbol"], funds["Scheme Name"]))
    
    print(f"Total unique symbols: {len(symbols)}")
    
    all_data = []
    
    # Download in batches
    for i, start_idx in enumerate(range(0, len(symbols), BATCH_SIZE), 1):
        batch = symbols[start_idx:start_idx + BATCH_SIZE]
        
        print(f"\n📦 Batch {i} | Processing {len(batch)} symbols")
        
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
            
            batch_rows = []
            
            for sym in batch:
                if sym not in raw:
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
            if i < len(range(0, len(symbols), BATCH_SIZE)):
                sleep_time = random.uniform(*SLEEP_RANGE)
                print(f"⏳ Cooling down for {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        except Exception as e:
            print(f"❌ Batch {i} failed: {e}")
            time.sleep(60)
    
    if not all_data:
        raise ValueError("No data downloaded!")
    
    data = pd.concat(all_data, ignore_index=True)
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
# STEP 3: CALCULATE TREND KPIs
# ========================
def calculate_trend_kpis(fy_data):
    """
    Calculate dividend yield trend KPIs:
    1. CAGR of dividend yield
    2. Linear trend slope (% increase per year)
    3. Consistency score (lower CV = more consistent)
    4. Consecutive years of increase
    5. YoY growth rates
    """
    print("📊 Calculating trend KPIs...")
    
    trend_kpis = []
    
    for (symbol, name), group in fy_data.groupby(["yahoo_symbol", "fund_name"]):
        # Sort by FY
        group = group.sort_values("FY").reset_index(drop=True)
        
        # Only calculate if we have at least 3 years of data
        if len(group) < 3:
            continue
        
        yields = group["IDCW_Yield_pct"].values
        years = np.arange(len(yields))
        
        # KPI 1: CAGR of dividend yield
        if yields[0] > 0 and yields[-1] > 0:
            n_years = len(yields) - 1
            cagr = ((yields[-1] / yields[0]) ** (1 / n_years) - 1) * 100
        else:
            cagr = None
        
        # KPI 2: Linear trend slope (regression)
        if len(yields) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, yields)
            trend_slope = slope  # % change per year
            trend_r_squared = r_value ** 2
        else:
            trend_slope = None
            trend_r_squared = None
        
        # KPI 3: Consistency (Coefficient of Variation)
        if yields.mean() > 0:
            cv = (yields.std() / yields.mean()) * 100
        else:
            cv = None
        
        # KPI 4: Consecutive years of increase
        increases = np.diff(yields) > 0
        max_consecutive = 0
        current_consecutive = 0
        
        for inc in increases:
            if inc:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        # KPI 5: Average YoY growth
        yoy_growth = np.diff(yields) / yields[:-1] * 100
        avg_yoy_growth = yoy_growth.mean() if len(yoy_growth) > 0 else None
        
        # Recent trend (last 3 years)
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
# STEP 4: EXTRACT SCHEME ATTRIBUTES
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
    
    # Load AMC list
    try:
        amc_list = pd.read_csv(amc_list_path)
    except FileNotFoundError:
        print("⚠️  AMC.csv not found. Creating default AMC list...")
        # Create a basic AMC list from fund names
        amc_list = pd.DataFrame({
            "AMC": [
                "Aditya Birla Sun Life Mutual Fund",
                "HDFC Mutual Fund",
                "ICICI Prudential Mutual Fund",
                "SBI Mutual Fund",
                "Axis Mutual Fund",
                "Kotak Mahindra Mutual Fund",
                "UTI Mutual Fund",
                "DSP Mutual Fund",
                "Nippon India Mutual Fund",
                "Franklin Templeton Mutual Fund",
                "Tata Mutual Fund",
                "HSBC Mutual Fund",
                "L&T Mutual Fund",
                "Mirae Asset Mutual Fund",
                "Motilal Oswal Mutual Fund",
                "Parag Parikh Mutual Fund",
                "Edelweiss Mutual Fund",
                "IDFC Mutual Fund",
                "Invesco Mutual Fund",
                "PPFAS Mutual Fund",
                "Quantum Mutual Fund",
                "Sundaram Mutual Fund",
                "Mahindra Manulife Mutual Fund",
                "Baroda BNP Paribas Mutual Fund",
                "Canara Robeco Mutual Fund",
                "360 ONE Mutual Fund"
            ]
        })
    
    # Clean columns
    amc_list["AMC_clean"] = amc_list["AMC"].apply(clean_text)
    funds_df["Scheme_clean"] = funds_df["Scheme Name"].apply(clean_text)
    
    # Build match keys
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
    
    # Match AMC
    def match_amc(s):
        match = compiled_pattern.search(s)
        return match.group(1) if match else "Unknown"
    
    funds_df["Matched_Key"] = funds_df["Scheme_clean"].apply(match_amc)
    
    # Build mapping dictionary
    mapping_dict = {}
    for _, row in amc_list.iterrows():
        mapping_dict[row["AMC_clean"]] = row["AMC"]
        first_word = row["AMC_clean"].split()[0]
        mapping_dict[first_word] = row["AMC"]
    
    funds_df["AMC"] = funds_df["Matched_Key"].map(mapping_dict).fillna("Unknown")
    
    # Extract Plan
    funds_df["Plan"] = "Unknown"
    funds_df.loc[funds_df["Scheme_clean"].str.contains(r"\bdirect\b", na=False), "Plan"] = "Direct"
    funds_df.loc[funds_df["Scheme_clean"].str.contains(r"\bregular\b", na=False), "Plan"] = "Regular"
    
    # Extract Option
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
    """Main execution function - generates all CSV files"""
    
    print("\n" + "="*70)
    print("MUTUAL FUND DATA GENERATOR")
    print("="*70 + "\n")
    
    try:
        # Step 1: Fetch and map ISIN to Yahoo
        print("STEP 1/4: Fetching AMFI data and mapping to Yahoo Finance")
        print("-" * 70)
        amfi = fetch_amfi_data()
        funds_with_yahoo = map_isins_to_yahoo(amfi)
        
        output_path = os.path.join(OUTPUT_DIR, "funds_mapped.csv")
        funds_with_yahoo.to_csv(output_path, index=False)
        print(f"💾 Saved: {output_path}")
        print(f"   Records: {len(funds_with_yahoo)}\n")
        
        # Step 2: Download performance data
        print("STEP 2/4: Downloading historical NAV and dividend data")
        print("-" * 70)
        performance_data = download_fund_performance(funds_with_yahoo)
        fy_metrics = calculate_fy_metrics(performance_data)
        
        output_path = os.path.join(OUTPUT_DIR, "idcw_fy_output.csv")
        fy_metrics.to_csv(output_path, index=False)
        print(f"💾 Saved: {output_path}")
        print(f"   Records: {len(fy_metrics)}\n")
        
        # Step 3: Calculate trend KPIs
        print("STEP 3/4: Calculating trend KPIs (CAGR, slope, consistency)")
        print("-" * 70)
        trend_kpis = calculate_trend_kpis(fy_metrics)
        
        output_path = os.path.join(OUTPUT_DIR, "trend_kpis.csv")
        trend_kpis.to_csv(output_path, index=False)
        print(f"💾 Saved: {output_path}")
        print(f"   Records: {len(trend_kpis)}\n")
        
        # Step 4: Extract scheme attributes
        print("STEP 4/4: Extracting scheme attributes (AMC, Plan, Option)")
        print("-" * 70)
        scheme_master = extract_scheme_attributes(funds_with_yahoo)
        
        output_path = os.path.join(OUTPUT_DIR, "final_schemes_master.csv")
        scheme_master.to_csv(output_path, index=False)
        print(f"💾 Saved: {output_path}")
        print(f"   Records: {len(scheme_master)}\n")
        
        # Summary
        print("=" * 70)
        print("✅ DATA GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nAll files saved to: {OUTPUT_DIR}/")
        print("\nGenerated files:")
        print(f"  1. funds_mapped.csv .............. {len(funds_with_yahoo):,} schemes")
        print(f"  2. idcw_fy_output.csv ............ {len(fy_metrics):,} fund-years")
        print(f"  3. trend_kpis.csv ................ {len(trend_kpis):,} funds")
        print(f"  4. final_schemes_master.csv ...... {len(scheme_master):,} schemes")
        print("\nNext step: Run the dashboard to visualize this data!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
