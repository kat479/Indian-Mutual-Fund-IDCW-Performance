# IDCW Fund Analyzer - Separated Architecture

A two-script system for analyzing Indian mutual funds by dividend yield trends.

## 📁 Project Structure

```
idcw-fund-analyzer/
├── data_generator.py          # Backend: Data collection & processing
├── dashboard.py                # Frontend: Streamlit visualization
├── requirements.txt            # Python dependencies
├── AMC.csv                     # (Optional) List of AMC names
├── data/                       # Generated data folder (auto-created)
│   ├── funds_mapped.csv
│   ├── idcw_fy_output.csv
│   ├── trend_kpis.csv
│   └── final_schemes_master.csv
└── README.md                   # This file
```

## 🎯 Two-Script Architecture

### Script 1: `data_generator.py` (Backend)
**Purpose**: Fetch, process, and save data - NO visualization

**What it does**:
- ✅ Fetches live AMFI mutual fund data
- ✅ Maps ISINs to Yahoo Finance symbols
- ✅ Downloads 8 years of historical NAV & dividend data
- ✅ Calculates FY-wise performance metrics
- ✅ Computes trend KPIs (CAGR, slope, consistency, etc.)
- ✅ Extracts scheme attributes (AMC, Plan, Option)
- ✅ Saves everything to CSV files in `data/` folder

**Outputs** (saved to `data/`):
1. `funds_mapped.csv` - AMFI data with Yahoo symbols
2. `idcw_fy_output.csv` - FY-wise NAV and dividend data
3. `trend_kpis.csv` - Calculated trend metrics
4. `final_schemes_master.csv` - Scheme attributes

**Run time**: 1-2 hours (one-time setup)

---

### Script 2: `dashboard.py` (Frontend)
**Purpose**: Visualize data from CSV files - NO data generation

**What it does**:
- ✅ Loads pre-generated CSV files from `data/` folder
- ✅ Displays interactive Streamlit dashboard
- ✅ Provides filtering, ranking, and sorting
- ✅ Shows fund deep-dive with charts
- ✅ Exports filtered results

**Requirements**: CSV files must exist in `data/` folder

**Run time**: Instant (loads pre-generated data)

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Generate Data (One-Time)

```bash
python data_generator.py
```

This will:
- Fetch AMFI data
- Map to Yahoo Finance
- Download historical data
- Calculate all metrics
- Save CSV files to `data/` folder

⏱️ **Time**: 1-2 hours (depending on network speed)

### Step 3: Launch Dashboard

```bash
streamlit run dashboard.py
```

Opens at: `http://localhost:8501`

⏱️ **Time**: Instant (loads from CSV files)

---

## 📊 Usage Workflow

### Regular Usage (After Initial Setup)

```bash
# Just run the dashboard
streamlit run dashboard.py
```

### Update Data (Weekly/Monthly)

```bash
# Re-run data generator
python data_generator.py

# Dashboard will auto-detect new data
# Or click "🔄 Refresh Data" in sidebar
```

---

## 🎨 Features

### Data Generator Features
- 📥 Fetches live AMFI data
- 🔍 Yahoo Finance symbol mapping (7,000+ funds)
- 📊 8 years of historical data
- 📈 Trend KPI calculations:
  - CAGR (Compound Annual Growth Rate)
  - Linear trend slope
  - Consistency (Coefficient of Variation)
  - Consecutive increases tracking
  - Year-over-year growth rates
- 🏷️ Automatic AMC/Plan/Option extraction
- 💾 Organized CSV output

### Dashboard Features
- 🎯 Multiple ranking methods:
  - Composite score (with trend)
  - Latest FY yield
  - Overall average yield
  - Best CAGR
  - Most consistent
- 🔍 Advanced filters:
  - Minimum years history
  - Minimum CAGR
  - Minimum consecutive increases
  - Positive trend only
  - AMC, Plan, Option filters
- 📊 Interactive charts:
  - NAV trends
  - Yield trends with trendlines
  - Year-over-year growth
  - Sparkline visualizations
- 💾 Export to CSV
- 🔄 Data refresh button

---

## 📁 Data Files Explained

### 1. `funds_mapped.csv`
Complete AMFI dataset with Yahoo Finance symbols
- All mutual fund schemes
- ISIN codes
- Yahoo symbols for data fetching
- Current NAV values

### 2. `idcw_fy_output.csv`
Financial year-wise performance data
- April NAV (start of FY)
- March NAV (end of FY)
- Total dividends paid
- FY returns
- IDCW yield percentage

### 3. `trend_kpis.csv`
Calculated trend metrics per fund
- CAGR of dividend yield
- Linear trend slope
- R-squared of trend
- Consistency (CV)
- Maximum consecutive increases
- Average YoY growth
- Recent 3-year trend

### 4. `final_schemes_master.csv`
Scheme attributes
- AMC (Asset Management Company)
- Plan (Direct/Regular)
- Option (Growth/IDCW)
- Scheme name

---

## 🔧 Configuration

### Modify Data Generation Settings

Edit `data_generator.py`:

```python
# Line 21-24
START_DATE = "2018-03-15"  # Start date for historical data
END_DATE = datetime.now()  # End date (defaults to today)
BATCH_SIZE = 300           # Funds per batch (Yahoo API)
SLEEP_RANGE = (15, 25)     # Sleep between batches (seconds)
```

### Customize AMC List

Create `AMC.csv` with your preferred AMC names:

```csv
AMC
Aditya Birla Sun Life Mutual Fund
HDFC Mutual Fund
ICICI Prudential Mutual Fund
SBI Mutual Fund
...
```

If not provided, script uses a default list.

---

## 🐛 Troubleshooting

### "Data directory not found"
**Solution**: Run `python data_generator.py` first

### "Missing required files"
**Solution**: Ensure data generator completed successfully. Check `data/` folder for all 4 CSV files.

### "No funds match your filters"
**Solution**: Relax filter criteria in dashboard sidebar

### Data generator taking too long
**Solution**: Normal! Downloading 7,000+ funds takes 1-2 hours. Progress is shown.

### Yahoo Finance rate limiting
**Solution**: Script includes automatic delays. If errors persist, increase `SLEEP_RANGE`.

---

## 📅 Recommended Update Frequency

- **Daily**: Not recommended (minimal changes)
- **Weekly**: Good for active investors
- **Monthly**: Sufficient for most users
- **Quarterly**: Minimum recommended

---

## 🚀 Deployment Options

### Option 1: Local Use (Recommended)
```bash
# Generate data locally
python data_generator.py

# Run dashboard locally
streamlit run dashboard.py
```

### Option 2: GitHub + Streamlit Cloud

**Structure**:
```
your-repo/
├── dashboard.py
├── requirements.txt
├── data/
│   └── (upload CSV files)
└── README.md
```

**Steps**:
1. Run `data_generator.py` locally
2. Upload generated `data/` folder to GitHub
3. Deploy `dashboard.py` to Streamlit Cloud
4. Update CSVs weekly/monthly as needed

**Advantages**:
- Free hosting on Streamlit Cloud
- No data generation on server (faster loading)
- Update data on your schedule

### Option 3: Automated Updates (Advanced)

Use GitHub Actions to auto-generate data weekly:

Create `.github/workflows/update_data.yml`:
```yaml
name: Update Fund Data
on:
  schedule:
    - cron: '0 0 * * 0'  # Every Sunday
  workflow_dispatch:

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python data_generator.py
      - run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add data/*.csv
          git commit -m "Auto-update fund data" || exit 0
          git push
```

---

## 💡 Pro Tips

### Faster Development
Run data generator once, then iterate on dashboard:
```bash
# One-time data generation
python data_generator.py

# Make changes to dashboard.py
# Test instantly
streamlit run dashboard.py
```

### Sample Data for Testing
Generate a small sample first:
```python
# In data_generator.py, line 163
# Limit to first 100 ISINs for testing
for i, isin in enumerate(amfi["ISIN"].unique()[:100]):
```

### Pre-filter During Generation
Only generate data for specific AMCs by filtering in `data_generator.py`:
```python
# After line 46
amfi = amfi[amfi["Scheme Name"].str.contains("HDFC|ICICI", case=False)]
```

---

## 📊 Data Statistics

Typical output sizes:
- **funds_mapped.csv**: ~14,000 rows (all schemes)
- **idcw_fy_output.csv**: ~50,000 rows (fund-years)
- **trend_kpis.csv**: ~3,000 rows (funds with 3+ years)
- **final_schemes_master.csv**: ~7,700 rows (unique schemes)

---

## ⚠️ Important Notes

1. **Data Generator is Heavy**: Downloads GBs of data from Yahoo Finance
2. **First Run Takes Time**: 1-2 hours is normal
3. **Dashboard is Lightweight**: Loads instantly from CSVs
4. **CSV Files are Portable**: Share `data/` folder with others
5. **Privacy**: All data is public (AMFI + Yahoo Finance)

---

## 📝 License

Open source - use freely for personal or educational purposes.

## ⚠️ Disclaimer

This tool is for informational purposes only. Not investment advice. Always verify data with official sources before making investment decisions.

---

## 🙏 Credits

- **Data Sources**: AMFI, Yahoo Finance
- **Libraries**: pandas, yfinance, streamlit, plotly, scipy

---

**Questions?** Open an issue on GitHub or check the troubleshooting section above.
