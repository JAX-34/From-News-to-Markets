import pandas as pd
from pathlib import Path

# ===== 路径与列名（按需改）=====
INFILE   = Path("newsoutfinal1.csv")   # 原始数据
DATECOL  = "date"             # 日期列名，例如 "2024/1/2"
OUT_TRAIN = Path("news_train.csv")
OUT_TEST  = Path("news_test.csv")

# ===== 月份范围（按需改）=====
TRAIN_START, TRAIN_END = "2024-01", "2024-09"  # 训练集：2024-01~2024-09
TEST_START,  TEST_END  = "2024-08", "2024-12"  # 测试集：2024-10~2024-12

# 读取并解析日期（支持 2024/1/2 这类格式；无法解析的置为 NaT）
df = pd.read_csv(INFILE, dtype=str)
df[DATECOL] = pd.to_datetime(df[DATECOL], errors="coerce")

# 丢弃无效日期并创建“年-月”Period列
bad = df[DATECOL].isna().sum()
df = df[df[DATECOL].notna()].copy()
df["ym"] = df[DATECOL].dt.to_period("M")

# 定义月份集合
train_months = pd.period_range(TRAIN_START, TRAIN_END, freq="M")
test_months  = pd.period_range(TEST_START,  TEST_END,  freq="M")

# 筛选、按日期排序并写出
train = df[df["ym"].isin(train_months)].drop(columns="ym").sort_values(DATECOL, kind="stable")
test  = df[df["ym"].isin(test_months)].drop(columns="ym").sort_values(DATECOL, kind="stable")

train.to_csv(OUT_TRAIN, index=False, encoding="utf-8-sig")
test.to_csv(OUT_TEST,  index=False, encoding="utf-8-sig")

print(f"Dropped invalid dates: {bad}")
print(f"Train rows: {len(train)}, Test rows: {len(test)}")
print(f"Saved to: {OUT_TRAIN.resolve()} and {OUT_TEST.resolve()}")
