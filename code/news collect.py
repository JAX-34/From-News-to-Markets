import os
import time
import re
import json
import math
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np

# key
API_KEY   = "personal key"#
STOCK_CSV = "test_stock_data.csv"
LANG      = "en"
COUNTRY   = "us"
OUT_DIR   = "news_out"
MAX_PER_PAGE = 100
PAGES_PER_WINDOW = 10


# Keyword mapping
QUERY_MAP = {
    "TSLA": '"Tesla" OR "Elon Musk"',
    "AAPL": '"Apple Inc." OR "Apple" OR "Apple stock"',
    "MSFT": '"Microsoft" OR "MSFT" OR "Azure"',
    "GOOG": '"Google" OR "Alphabet Inc." OR "Waymo"',
    "AMZN": '"Amazon.com" OR "Amazon" OR "AWS"',
    "META": '"Meta" OR "Facebook" OR "Instagram" OR "WhatsApp"',
    "NVDA": '"NVIDIA" OR "NVDA" OR "GPU"',
    "NIO" : '"NIO" OR "蔚来" OR "蔚来汽车"',
    "RIVN": '"Rivian"'
}

#  FinBERT (financial sentiment)
from transformers import BertTokenizer, BertForSequenceClassification
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone').to(DEVICE)
model.eval()

def finbert_score(text: str):

    text = (text or "").strip()
    if not text:
        return "neutral", 0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    idx = probs.argmax()
    label_map = {0: "positive", 1: "neutral", 2: "negative"}  # 注意：该模型类目顺序是 positive, neutral, negative
    label = label_map[idx]
    score = {"positive": 1, "neutral": 0, "negative": -1}[label]
    return label, score

# ======== 工具函数 ========
def month_spans(start_date: pd.Timestamp, end_date: pd.Timestamp):
    """把整体日期拆成若干自然月区间（字符串 YYYY-MM-DD）"""
    spans = []
    cur = pd.Timestamp(year=start_date.year, month=start_date.month, day=1)
    end_month_start = pd.Timestamp(year=end_date.year, month=end_date.month, day=1)
    while cur <= end_month_start:
        if cur.month == 12:
            month_end = pd.Timestamp(year=cur.year, month=12, day=31)
        else:
            month_end = (cur + pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)
        # 裁剪到用户范围
        s = max(cur, start_date)
        e = min(month_end, end_date)
        spans.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
        cur = cur + pd.offsets.MonthBegin(1)
    return spans

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # 去不可见/非ASCII（Excel常见“鈥”类）
    s = re.sub(r'[^\x00-\x7F]+', ' ', s)
    # 压缩多空格
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def gnews_search(query, start_date, end_date, page=1, max_results=MAX_PER_PAGE):
    """调用 GNews 会员接口（支持 from/to/page）"""
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": LANG,
        "country": COUNTRY,
        "from": start_date,
        "to": end_date,
        "max": max_results,
        "page": page,
        "apikey": API_KEY
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        print(f"❌ GNews {resp.status_code}: {resp.text[:200]}")
        return None
    return resp.json()

def collect_company_month(ticker, query, start_date, end_date):
    """抓取单公司某个月份所有页，返回 list[dict]"""
    all_items = []
    for p in range(1, PAGES_PER_WINDOW + 1):
        data = gnews_search(query, start_date, end_date, page=p)
        if not data:
            break
        arts = data.get("articles", [])
        if not arts:
            break
        for a in arts:
            all_items.append({
                "date": (a.get("publishedAt","")[:10] or ""),
                "ticker": ticker,
                "title": clean_text(a.get("title","")),
                "description": clean_text(a.get("description","")),
                "content": clean_text(a.get("content","")),
                "source": (a.get("source",{}) or {}).get("name",""),
                "url": a.get("url",""),
                "query": query
            })
        # 简单限频
        time.sleep(0.6)
    return all_items

# ======== 主流程 ========
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 读取股票训练集，确定日期范围与 Ticker 列
    stock = pd.read_csv(STOCK_CSV)
    if "Date" not in stock.columns:
        raise ValueError("股票CSV缺少 Date 列")
    stock["Date"] = pd.to_datetime(stock["Date"])
    date_min = stock["Date"].min()
    date_max = stock["Date"].max()
    tickers = [c for c in stock.columns if c != "Date"]
    print(f"✅ 读取股票表：{STOCK_CSV}，日期范围 {date_min.date()} ~ {date_max.date()}，股票：{tickers}")

    # 2) 逐公司逐月份抓新闻
    raw_rows = []
    spans = month_spans(date_min, date_max)
    for tkr in tickers:
        query = QUERY_MAP.get(tkr, f'"{tkr}" stock OR earnings')
        print(f"\n==== {tkr} | query: {query} ====")
        for s, e in spans:
            print(f"  ⌛ 抓取 {s} ~ {e}")
            items = collect_company_month(tkr, query, s, e)
            print(f"    ➕ 本月抓到 {len(items)} 条")
            raw_rows.extend(items)

    if not raw_rows:
        print("⚠️ 未抓到任何新闻，请检查 API Key / 配额 / 参数")
        return

    raw_df = pd.DataFrame(raw_rows)
    raw_df = raw_df.drop_duplicates(subset=["date","ticker","title"]).reset_index(drop=True)

    # 3) 运行 FinBERT 情绪
    print("\n🧠 FinBERT 情绪分析中…")
    labels = []
    scores = []
    for i, row in raw_df.iterrows():
        text = row["title"] if row["title"] else row["description"]
        label, score = finbert_score(text)
        labels.append(label)
        scores.append(score)
        if (i+1) % 200 == 0:
            print(f"  已处理 {i+1}/{len(raw_df)}")
    raw_df["sentiment_label"] = labels
    raw_df["sentiment_score"] = scores

    # 4) 保存逐条新闻明细
    year_tag = f"{date_min.year}" if date_min.year == date_max.year else f"{date_min.year}-{date_max.year}"
    raw_path = os.path.join(OUT_DIR, f"news_raw_{year_tag}.csv")
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"✅ 明细已保存：{raw_path}（{len(raw_df)}条）")

    # 5) 聚合成“日×股票”的宽表，与股票日期对齐
    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
    daily = (
        raw_df.groupby(["date","ticker"])
              .agg(avg_sentiment=("sentiment_score","mean"),
                   news_count=("sentiment_score","count"))
              .reset_index()
    )

    # 生成与股票表同样的日期索引
    all_days = stock[["Date"]].drop_duplicates().sort_values("Date")
    # pivot 平均情绪
    mat_sent = daily.pivot(index="date", columns="ticker", values="avg_sentiment").reindex(all_days["Date"]).reset_index()
    mat_sent.rename(columns={"date":"Date"}, inplace=True)
    # 缺失填 0（当天无新闻）
    for t in tickers:
        if t not in mat_sent.columns:
            mat_sent[t] = 0.0
    mat_sent = mat_sent[["Date"] + tickers].fillna(0.0)

    # pivot 新闻条数
    mat_cnt = daily.pivot(index="date", columns="ticker", values="news_count").reindex(all_days["Date"]).reset_index()
    mat_cnt.rename(columns={"date":"Date"}, inplace=True)
    for t in tickers:
        if t not in mat_cnt.columns:
            mat_cnt[t] = 0
    mat_cnt = mat_cnt[["Date"] + tickers].fillna(0).astype({t:int for t in tickers})

    # 6) 保存宽表（与你股票表结构一致）
    sent_path = os.path.join(OUT_DIR, "news_daily_sentiment.csv")
    cnt_path  = os.path.join(OUT_DIR, "news_daily_count.csv")
    mat_sent.to_csv(sent_path, index=False, encoding="utf-8-sig")
    mat_cnt.to_csv(cnt_path,  index=False, encoding="utf-8-sig")

    print(f"✅ 日均情绪矩阵：{sent_path}  形状={mat_sent.shape}")
    print(f"✅ 当日新闻条数：{cnt_path}   形状={mat_cnt.shape}")
    print("\n🎯 现在你已经拥有：")
    print("   1) 与股票表同结构的“日均情绪矩阵”（可直接并列拼接）")
    print("   2) 当日新闻条数矩阵（可作权重或特征）")
    print("   3) 全量逐条新闻明细（可复核与溯源）")

if __name__ == "__main__":
    main()
