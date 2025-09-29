From-News-to-Markets

Modeling US stock direction from news and prices (Ridge / XGBoost / lightweight Transformer).
Universe: 9 tickers (AAPL, AMZN, GOOG, META, MSFT, NVDA, TSLA, NIO, RIVN) in 2024-01 ~ 2024-12.
Due to sparse coverage for NIO and RIVN, the final modeling set keeps 7 large-cap stocks.
Time split to avoid look-ahead: 2024-01~09 for train/val (the last 30 trading days of September for validation), 2024-10~12 for test.

API Key & Data Notice

The GNews API key is personally purchased and will not be shared. The repository therefore does not include a usable key in the news collection code.

The committed news datasets are real and reliable and can be used directly for alignment, feature engineering, and modeling.

If you want to re-collect news from scratch, create a local .env (do not commit it):

GNEWS_TOKEN=YOUR_GNEWS_TOKEN_HERE

and read it in code (already supported or add as needed):

import os
from dotenv import load_dotenv
load_dotenv()
KEY = os.getenv("GNEWS_TOKEN", "")

**Project Layout**

```text
.                                # From-News-to-Markets (repo root)
├─ 3model.py                     # Training & evaluation: Ridge / XGB / lightweight Transformer
├─ data partitioning.py          # Time split & alignment: produce train/val/test
├─ news collect.py               # News scraping/cleaning (requires local GNEWS_TOKEN)
├─ stock price.py                # Daily price download & prep (Yahoo Finance)
└─ data/                         # Data files (recommend .gitignore or Git LFS)
   ├─ newsoutfinal1.csv          # Cleaned news merged output
   ├─ news_test.csv              # Test news
   ├─ news_train.csv             # Train/val news
   ├─ stock_test.csv             # Test prices
   ├─ stock_train.csv            # Train/val prices
   └─ test_stock_data.csv        # Intermediate/sample artifact


**Create environment & install deps

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt   # if missing, install packages based on script imports

(Optional) 
python "stock price.py"
python "news collect.py"
python "data partitioning.py"
python "3model.py"




