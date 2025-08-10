# 🚀 Quick Start: Complete 4H Data Collection

## 📋 Pre-flight Checklist

1. **Check Current Status**
   ```bash
   # See which tokens need 4H data
   node tools/analyzers/collection-stats.js
   ```

2. **Estimate Time & Resources**
   - **Tokens to collect**: 59
   - **Estimated time**: 4-5 hours
   - **API calls**: ~2,360
   - **Storage needed**: ~50MB

## 🏃 Run Collection

### Option 1: Run Full Collection (Recommended)
```bash
# This will collect 4H data for all 59 missing tokens
node tools/collectors/complete-all-4h-collection.js
```

### Option 2: Test with Few Tokens First
```bash
# Edit the script to limit tokens (change line 36)
# symbolLimit: 5  // Test with 5 tokens first
nano tools/collectors/complete-all-4h-collection.js

# Then run
node tools/collectors/complete-all-4h-collection.js
```

## 📊 Monitor Progress

The script will show:
- Real-time progress for each token
- Success/failure status
- Estimated time remaining
- Final summary report

Example output:
```
🚀 Starting 4H data collection for missing tokens
📊 Total tokens to process: 59
⚙️  Configuration: { timeframe: '4H', startDate: '2021-01-01', ... }

📊 Collecting 4H data for AAVE (AAVE-USDT)
  📈 Batch 1: 300 candles (total: 300)
  📈 Batch 2: 300 candles (total: 600)
  ...
✅ AAVE: Saved 8500 candles

📊 Progress: 1/59 (2%)
⏱️  Elapsed: 4.2min | Rate: 0.2/min | ETA: 280min
```

## 🛠️ Troubleshooting

### If Collection Fails
1. The script automatically retries failed requests (3 times)
2. Failed tokens are reported at the end
3. You can re-run the script - it skips already collected data

### Rate Limiting
- Script handles OKX rate limits automatically
- 500ms delay between requests
- 5 concurrent collections maximum

### Resume After Interruption
Simply run the script again:
```bash
node tools/collectors/complete-all-4h-collection.js
```
It will skip tokens that already have 4H data.

## ✅ Verify Results

After completion:
```bash
# Check collection report
cat data/okx/metadata/4h-collection-report.json

# Verify specific token
node tools/utils/data-reader.js read data/okx/AAVE/4H/data.json

# Count tokens with 4H data
find data/okx -name "data.json" -path "*/4H/*" | wc -l
```

## 🔄 Next Steps

1. **Keep Data Updated**
   ```bash
   # Run daily to get latest candles
   node tools/collectors/incremental-updater.js
   ```

2. **Compress Data (Optional)**
   ```bash
   # Reduce storage by 63%
   node tools/utils/batch-compress.js compress --strategy optimized
   ```

3. **Add More Timeframes**
   ```bash
   # Add 1H data for top tokens
   node tools/collectors/multi-timeframe-collector.js BTC ETH SOL --timeframes 1H
   ```

---

**Ready to start?** Just run:
```bash
node tools/collectors/complete-all-4h-collection.js
```

The script is robust and handles errors gracefully. You can safely let it run unattended.