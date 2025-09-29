import warnings, argparse, math, gc
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# graph save or show
SHOW_FIGS = True   # True=show；False=save

# pass
NEWS_TRAIN_CSV  = "news_train.csv"
NEWS_TEST_CSV   = "news_test.csv"
STOCK_TRAIN_CSV = "stock_train.csv"
STOCK_TEST_CSV  = "stock_test.csv"

# major hyperparameter
COVERAGE_THRESHOLD = 0.10    # Minimum news coverage for stocks participating in training/assessment
VAL_DAYS           = 30      # The last N days of the training set are used as validation
HORIZON_DAYS       = 5       # Predict the 5-day direction
MIN_ABS_EXCESS     = 0.0015  # Weak signal removal threshold (excess return relative to the market)
HALF_LIFE_DAYS     = 40      # Time decay half-life
LOOKBACK           = 30      # Transformer sequence length
BATCH_SIZE         = 64
EPOCHS             = 60
PATIENCE           = 8
SEED               = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# Column name cleaning, column matching, date standardization, interval statistics.
def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff","").strip() for c in df.columns]
    return df

def _find_col(df: pd.DataFrame, cands_lower):
    mapping = {str(c).replace("\ufeff","").strip().lower(): c for c in df.columns}
    for k in cands_lower:
        if k in mapping: return mapping[k]
    return None

def ensure_datetime(s):
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()

def days_since_nonzero(arr: np.ndarray):
    out = np.empty_like(arr, dtype=float); last=None
    for i,v in enumerate(arr):
        if v>0: out[i]=0.0; last=0
        else:
            if last is None: out[i]=np.nan
            else: last+=1; out[i]=float(last)
    return out

def time_decay_weight(dates: np.ndarray, end_date: pd.Timestamp, half_life_days=60):
    days = (end_date - pd.to_datetime(dates)).days.astype(float)
    return np.power(0.5, days / max(1.0, half_life_days)).astype("float32")


#  Compatible with news CSV from different sources. Automatically clean column names and in multiple aliases
def read_news_any(path: str) -> pd.DataFrame:
    df = _std_cols(pd.read_csv(path, engine="python", encoding="utf-8-sig"))
    c_date = _find_col(df, ["date","datetime","time","timestamp","publishedat","published_at","pub_date","pubdate","时间","日期"])
    c_tick = _find_col(df, ["ticker","symbol","code","stock","company"])
    if c_date is None or c_tick is None:
        raise KeyError(f"[news] 缺少 date/ticker: {list(df.columns)}")
    c_sent = _find_col(df, ["sentiment_score","sent_score","score","compound","polarity","finbert_score"])
    c_pos  = _find_col(df, ["p_positive","positive","pos","prob_pos"])
    c_neg  = _find_col(df, ["p_negative","negative","neg","prob_neg"])
    out = pd.DataFrame({
        "ticker": df[c_tick].astype(str).str.upper().str.strip(),
        "date":   ensure_datetime(df[c_date]),
        "sentiment_score": pd.to_numeric(df[c_sent], errors="coerce") if c_sent else 0.0,
        "p_positive": pd.to_numeric(df[c_pos], errors="coerce") if c_pos else np.nan,
        "p_negative": pd.to_numeric(df[c_neg], errors="coerce") if c_neg else np.nan,
    }).dropna(subset=["date"])
    out["p_positive"] = out["p_positive"].fillna((out["sentiment_score"]>0).astype(float))
    out["p_negative"] = out["p_negative"].fillna((out["sentiment_score"]<0).astype(float))
    return out
# Compatible with "long tables (ticker,date,close)" and "wide tables (date, each ticker column)"
def read_prices_any(path: str) -> pd.DataFrame:

    df = _std_cols(pd.read_csv(path, engine="python", encoding="utf-8-sig"))
    if all(k in [c.lower() for c in df.columns] for k in ["ticker","date","close"]):
        c_tk = _find_col(df, ["ticker"]); c_dt = _find_col(df, ["date"]); c_cl = _find_col(df, ["close"])
        out = pd.DataFrame({
            "ticker": df[c_tk].astype(str).str.upper().str.strip(),
            "date":   ensure_datetime(df[c_dt]),
            "close":  pd.to_numeric(df[c_cl], errors="coerce"),
        }).dropna(subset=["date","close"])
    else:
        c_date = _find_col(df, ["date","datetime","trade_date","timestamp","time","时间","日期"])
        if c_date is None: raise KeyError(f"[prices] 缺少日期列: {list(df.columns)}")
        df["date"] = ensure_datetime(df[c_date])
        out = df.melt(id_vars=["date"], var_name="ticker", value_name="close")
        out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
        out["close"]  = pd.to_numeric(out["close"], errors="coerce")
        out = out.dropna(subset=["date","close"])

    #  Reduce the repetition of the same ticket on the same day
    out = (out.sort_values(["ticker","date"])
              .groupby(["ticker","date"], as_index=False, sort=True)["close"]
              .last())
    return out.reset_index(drop=True)


# Price characteristics
def add_price_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker","date"]).reset_index(drop=True).copy()
    df["ret_1d"]  = df.groupby("ticker")["close"].transform(lambda x: np.log(x).diff())
    df["ret_5d"]  = df.groupby("ticker")["close"].transform(lambda x: np.log(x).diff(5))
    df["ret_10d"] = df.groupby("ticker")["close"].transform(lambda x: np.log(x).diff(10))
    df["vol_10d"] = df.groupby("ticker")["ret_1d"].transform(lambda x: x.rolling(10, min_periods=2).std())
    df["ma5"]     = df.groupby("ticker")["close"].transform(lambda x: x.rolling(5,  min_periods=1).mean())
    df["ma20"]    = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df["ma_gap"]  = df["ma5"] - df["ma20"]

    def rsi(series, n=14):
        d = series.diff(); up = d.clip(lower=0); down = (-d.clip(upper=0))
        up = up.rolling(n, min_periods=1).mean(); down = down.rolling(n, min_periods=1).mean()
        rs = up/(down+1e-8); return 100 - (100/(1+rs))
    df["rsi14"] = df.groupby("ticker")["close"].transform(lambda x: rsi(x, 14))

    H = HORIZON_DAYS
    df["close_fh"] = df.groupby("ticker")["close"].shift(-H)
    df["y_reg_h"]  = np.log(df["close_fh"]) - np.log(df["close"])
    return df


# main process: read + feature
def load_and_featurize():
    news_tr  = read_news_any(NEWS_TRAIN_CSV)
    news_te  = read_news_any(NEWS_TEST_CSV)
    prices_tr = read_prices_any(STOCK_TRAIN_CSV)
    prices_te = read_prices_any(STOCK_TEST_CSV)

    def agg_news(n):
        return (n.groupby(["ticker","date"])
                  .agg(sent_mean=("sentiment_score","mean"),
                       sent_pos_mean=("p_positive","mean"),
                       sent_neg_mean=("p_negative","mean"),
                       n_news=("sentiment_score","size"))
                  .reset_index())
    daily_news = pd.concat([agg_news(news_tr), agg_news(news_te)], ignore_index=True)

    prices = pd.concat([prices_tr, prices_te], ignore_index=True).dropna(subset=["close"])
    prices = prices.sort_values(["ticker","date"]).reset_index(drop=True)
    prices = add_price_feats(prices)

    train_start, train_end = prices_tr["date"].min(), prices_tr["date"].max()
    test_start,  test_end  = prices_te["date"].min(), prices_te["date"].max()
    val_start = sorted(prices_tr["date"].unique())[-VAL_DAYS]
    print(f"Train: {train_start.date()}~{train_end.date()} | Val: {val_start.date()}~{train_end.date()} | Test: {test_start.date()}~{test_end.date()}")

    # Coverage rate screening stocks
    coverage = {}
    for tk,g in prices.groupby("ticker"):
        d_price = set(g["date"].unique())
        d_news  = set(daily_news[daily_news["ticker"]==tk]["date"].unique())
        coverage[tk] = len(d_price & d_news) / max(1, len(d_price))
    selected = [tk for tk,cv in coverage.items() if cv>=COVERAGE_THRESHOLD] or sorted(prices["ticker"].unique())
    print("Stocks used for training/evaluation：", selected)

    df = pd.merge(prices, daily_news, on=["ticker","date"], how="left").sort_values(["ticker","date"]).reset_index(drop=True)
    for c in ["sent_mean","sent_pos_mean","sent_neg_mean","n_news"]:
        df[c] = df[c].fillna(0.0)

    # Lag avoids leakage & simple derivation
    for c in ["sent_mean","sent_pos_mean","sent_neg_mean","n_news"]:
        df[c] = df.groupby("ticker")[c].shift(1)
    for k in [2,3]:
        for c in ["sent_mean","sent_pos_mean","sent_neg_mean","n_news"]:
            df[f"{c}_l{k}"] = df.groupby("ticker")[c].shift(k)

    for c in ["sent_mean","n_news"]:
        df[f"{c}_7d_mean"] = df.groupby("ticker")[c].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df[f"{c}_7d_std"]  = df.groupby("ticker")[c].transform(lambda x: x.rolling(7, min_periods=2).std())
        df[f"{c}_7d_z"]    = (df[c] - df[f"{c}_7d_mean"]) / (df[f"{c}_7d_std"] + 1e-8)

    df["has_news_1d"]     = (df["n_news"] > 0).astype(float)
    df["days_since_news"] = df.groupby("ticker")["n_news"].transform(lambda x: pd.Series(days_since_nonzero(x.values))).fillna(99.0)
    df["bull"]         = df["sent_pos_mean"] - df["sent_neg_mean"]
    df["sent_abs"]     = df["sent_mean"].abs()
    df["n_news_log"]   = np.log1p(df["n_news"])
    df["sent3_decay"]  = 0.6*df["sent_mean"] + 0.3*df["sent_mean_l2"] + 0.1*df["sent_mean_l3"]
    df["news_burst"]   = df["n_news"] / (df["n_news_7d_mean"] + 1e-6)
    df["sent_dev"]     = df["sent_mean"] - df["sent_mean_7d_mean"]
    df["sent_flip"]    = (np.sign(df["sent_mean"]) != np.sign(df["sent_mean_l2"].fillna(0))).astype(float)

    dow = df["date"].dt.weekday
    df["dow_sin"] = np.sin(2*np.pi*dow/7); df["dow_cos"] = np.cos(2*np.pi*dow/7)
    df["vol_20d"] = df.groupby("ticker")["ret_1d"].transform(lambda x: x.rolling(20, min_periods=3).std())

    def slope5(x):
        if len(x)<5: return np.nan
        xi = np.arange(5); y = np.log(x[-5:]); return np.polyfit(xi,y,1)[0]
    df["mom_slope_5d"] = df.groupby("ticker")["close"].transform(lambda s: s.rolling(5).apply(slope5, raw=False))

    # Market equal weight excess (pivot_table is more stable)
    prices_daily = (pd.concat([prices_tr, prices_te], ignore_index=True)
                    .groupby(["date","ticker"], as_index=False)
                    .agg(close=("close","last")))
    wide = prices_daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    eq = wide.mean(axis=1).rename("mkt_close").to_frame()
    eq["mkt_fh"]     = eq["mkt_close"].shift(-HORIZON_DAYS)
    eq["mkt_ret_h"]  = np.log(eq["mkt_fh"]) - np.log(eq["mkt_close"])
    eq["mkt_ret_1d"] = np.log(eq["mkt_close"]).diff()
    df = df.merge(eq[["mkt_ret_1d","mkt_ret_h"]], left_on="date", right_index=True, how="left")

    df["y_reg_excess"] = df["y_reg_h"] - df["mkt_ret_h"]
    df["y_cls"] = (df["y_reg_excess"] > 0).astype(int)

    base_cols = [
        "ret_1d","ret_5d","ret_10d","vol_10d","vol_20d","ma_gap","rsi14","mom_slope_5d",
        "mkt_ret_1d",
        "sent3_decay","bull","sent_abs","n_news_log","has_news_1d","days_since_news",
        "news_burst","sent_dev","sent_flip",
        "sent_mean_7d_mean","sent_mean_7d_std","sent_mean_7d_z",
        "n_news_7d_mean","n_news_7d_std","n_news_7d_z",
        "dow_sin","dow_cos"
    ]
    for c in ["ret_1d","ret_5d","ret_10d","vol_10d","ma_gap","rsi14","sent_abs","n_news_log","sent3_decay","bull","vol_20d","mom_slope_5d"]:
        df[c+"_pct"] = df.groupby("date")[c].rank(pct=True)
        base_cols.append(c+"_pct")

    df = df.dropna(subset=base_cols + ["y_cls"]).reset_index(drop=True)
    #Split the collection strictly by the date window
    is_train_all = (df["date"]>=train_start) & (df["date"]<=train_end)
    is_val       = (df["date"]>=val_start) & (df["date"]<=train_end)
    is_train     = is_train_all & (~is_val)
    is_test      = (df["date"]>=test_start)  & (df["date"]<=test_end)

    mask_selected = df["ticker"].isin(selected)
    is_train &= mask_selected; is_val &= mask_selected; is_test &= mask_selected

    print("Sample size | train=%d | val=%d | test=%d" % (int(is_train.sum()), int(is_val.sum()), int(is_test.sum())))

    # 目标编码 + 去弱信号
    g = df.loc[is_train, ["ticker","y_cls"]].groupby("ticker")["y_cls"].agg(["sum","count"])
    global_mean = df.loc[is_train, "y_cls"].mean() if is_train.any() else 0.5
    m = 10.0
    g["te"] = (g["sum"] + m*global_mean) / (g["count"] + m)
    df["tk_te"] = df["ticker"].map(g["te"].to_dict()).fillna(global_mean)

    keep = np.ones(len(df), dtype=bool)
    keep[is_train] = (df.loc[is_train, "y_reg_excess"].abs().values >= MIN_ABS_EXCESS)
    is_train = is_train & keep

    return df, base_cols, is_train, is_val, is_test


# XGBoost
def train_xgb(df, feature_cols, is_train, is_val, is_test):
    X_train = df.loc[is_train, feature_cols + ["tk_te"]].values
    y_train = df.loc[is_train,"y_cls"].values
    X_val   = df.loc[is_val,   feature_cols + ["tk_te"]].values
    y_val   = df.loc[is_val,  "y_cls"].values
    X_test  = df.loc[is_test,  feature_cols + ["tk_te"]].values
    y_test  = df.loc[is_test, "y_cls"].values

    w_train = time_decay_weight(df.loc[is_train,"date"].values, df.loc[is_train,"date"].max(), HALF_LIFE_DAYS)
    spw = max(1, int((y_train==0).sum())) / max(1, int((y_train==1).sum()))

    params = dict(
        objective="binary:logistic", eval_metric="auc", eta=0.03,
        max_depth=5, min_child_weight=2.0, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=4.0, base_score=0.5, scale_pos_weight=float(spw),
        tree_method="hist", grow_policy="lossguide", max_leaves=128, seed=SEED
    )
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, feature_names=feature_cols+["tk_te"])
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_cols+["tk_te"])
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feature_cols+["tk_te"])
    booster = xgb.train(params, dtrain, num_boost_round=4000,
                        evals=[(dval,"val")], early_stopping_rounds=200, verbose_eval=False)

    def pred(dm):
        try:
            p = booster.predict(dm, iteration_range=(0, booster.best_iteration+1))
        except Exception:
            p = booster.predict(dm, ntree_limit=getattr(booster,"best_ntree_limit",0))
        return p

    val_p  = pred(dval); test_p = pred(dtest)
    best_thr, best_acc = 0.5, -1
    for thr in np.linspace(0.3,0.7,41):
        acc = accuracy_score(y_val, (val_p>=thr).astype(int))
        if acc>best_acc: best_acc, best_thr=acc, thr
    val_auc  = roc_auc_score(y_val, val_p) if len(np.unique(y_val))>1 else float("nan")
    test_auc = roc_auc_score(y_test, test_p) if len(np.unique(y_test))>1 else float("nan")
    test_acc = accuracy_score(y_test, (test_p>=best_thr).astype(int))
    print(f"[XGB]   Val AUC={val_auc:.3f} | thr={best_thr:.2f} (Acc={best_acc:.3f})")
    print(f"[XGB]   Test AUC={test_auc:.3f} | Acc(thr={best_thr:.2f})={test_acc:.3f}")

    booster.save_model("xgb_model.json")
    pd.DataFrame({"p":val_p}).to_csv("xgb_val_probs.csv", index=False)
    pd.DataFrame({"p":test_p}).to_csv("xgb_test_probs.csv", index=False)

    return {"name":"xgb","val_p":val_p,"test_p":test_p,"y_val":y_val,"y_test":y_test,"thr":best_thr,
            "val_auc":val_auc,"test_auc":test_auc,"booster":booster}


# Ridge Logistic
def train_ridge(df, feature_cols, is_train, is_val, is_test):
    feats = feature_cols + ["tk_te"]
    X_train = df.loc[is_train, feats].values; y_train = df.loc[is_train,"y_cls"].values.astype(np.float32)
    X_val   = df.loc[is_val,   feats].values; y_val   = df.loc[is_val,  "y_cls"].values.astype(np.float32)
    X_test  = df.loc[is_test,  feats].values; y_test  = df.loc[is_test, "y_cls"].values.astype(np.float32)
    w_train = time_decay_weight(df.loc[is_train,"date"].values, df.loc[is_train,"date"].max(), HALF_LIFE_DAYS)

    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train); X_val_sc = scaler.transform(X_val); X_test_sc = scaler.transform(X_test)

    best_auc, best_C, best_model = -1, 1.0, None
    for C in [0.01,0.05,0.1,0.5,1.0,2.0,5.0,10.0]:
        lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000, C=C)
        lr.fit(X_train_sc, y_train, sample_weight=w_train)
        val_p = lr.predict_proba(X_val_sc)[:,1]
        auc = roc_auc_score(y_val, val_p) if len(np.unique(y_val))>1 else float("nan")
        if auc>best_auc: best_auc, best_C, best_model = auc, C, lr
    val_p  = best_model.predict_proba(X_val_sc)[:,1]
    test_p = best_model.predict_proba(X_test_sc)[:,1]
    best_thr, best_acc = 0.5, -1
    for thr in np.linspace(0.3,0.7,41):
        acc = accuracy_score(y_val, (val_p>=thr).astype(int))
        if acc>best_acc: best_acc, best_thr=acc, thr
    val_auc  = roc_auc_score(y_val, val_p) if len(np.unique(y_val))>1 else float("nan")
    test_auc = roc_auc_score(y_test, test_p) if len(np.unique(y_test))>1 else float("nan")
    test_acc = accuracy_score(y_test, (test_p>=best_thr).astype(int))
    print(f"[RIDGE] Val AUC={val_auc:.3f} (C={best_C}) | thr={best_thr:.2f} (Acc={best_acc:.3f})")
    print(f"[RIDGE] Test AUC={test_auc:.3f} | Acc(thr={best_thr:.2f})={test_acc:.3f}")

    pd.DataFrame({"p":val_p}).to_csv("ridge_val_probs.csv", index=False)
    pd.DataFrame({"p":test_p}).to_csv("ridge_test_probs.csv", index=False)

    return {"name":"ridge","val_p":val_p,"test_p":test_p,"y_val":y_val,"y_test":y_test,"thr":best_thr,"val_auc":val_auc,"test_auc":test_auc}


# Transformer
class SeqDataset(Dataset):
    def __init__(self, Xseq, y, w):
        self.Xseq = Xseq.astype(np.float32); self.y=y.astype(np.float32); self.w=w.astype(np.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.Xseq[idx], self.y[idx], self.w[idx]

class TrfModel(nn.Module):
    def __init__(self, d_feat, nhead=4, nlayers=2, d_ff=128, p=0.2):
        super().__init__()
        self.proj  = nn.Linear(d_feat, d_ff)
        enc_layer  = nn.TransformerEncoderLayer(d_model=d_ff, nhead=nhead, dim_feedforward=4*d_ff,
                                                dropout=p, batch_first=True, activation='gelu')
        self.enc   = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head  = nn.Sequential(nn.LayerNorm(d_ff), nn.Linear(d_ff, 1))
    def forward(self, x):
        h = self.proj(x); h = self.enc(h); h = h[:, -1, :]
        return self.head(h).squeeze(-1)

def make_sequences(df, feature_cols, is_mask):
    feats = feature_cols + ["tk_te"]
    sub = df.loc[is_mask, ["ticker","date","y_cls"] + feats].copy()
    sub = sub.sort_values(["ticker","date"]).reset_index(drop=True)
    X_list, y_list, date_list = [], [], []
    for tk, g in sub.groupby("ticker"):
        g = g.reset_index(drop=True)
        arr = g[feats].values; y = g["y_cls"].values; dates = g["date"].values
        if len(g) < LOOKBACK: continue
        for i in range(LOOKBACK-1, len(g)):
            X_list.append(arr[i-LOOKBACK+1:i+1, :])
            y_list.append(y[i]); date_list.append(dates[i])
    X = np.array(X_list); y = np.array(y_list); dates = np.array(date_list)
    return X, y, dates

def train_transformer(df, feature_cols, is_train, is_val, is_test):
    Xtr, ytr, dtr = make_sequences(df, feature_cols, is_train)
    Xva, yva, dva = make_sequences(df, feature_cols, is_val)
    Xte, yte, dte = make_sequences(df, feature_cols, is_test)

    if len(ytr)==0 or len(yva)==0 or len(yte)==0:
        print("[TRF] 序列样本不足，跳过 Transformer。")
        return {"name":"trf","val_p":np.array([]),"test_p":np.array([]),"y_val":np.array([]),"y_test":np.array([]),
                "thr":0.5,"val_auc":float("nan"),"test_auc":float("nan")}

    F = Xtr.shape[-1]
    scaler = StandardScaler().fit(Xtr.reshape(-1, F))
    Xtr = scaler.transform(Xtr.reshape(-1, F)).reshape(Xtr.shape)
    Xva = scaler.transform(Xva.reshape(-1, F)).reshape(Xva.shape)
    Xte = scaler.transform(Xte.reshape(-1, F)).reshape(Xte.shape)

    wtr = time_decay_weight(dtr, dtr.max(), HALF_LIFE_DAYS)
    pos = max(1, int((ytr==1).sum())); neg = max(1, int((ytr==0).sum()))
    pos_weight = torch.tensor([neg/pos], dtype=torch.float32)

    train_ds = SeqDataset(Xtr, ytr, wtr); val_ds = SeqDataset(Xva, yva, np.ones_like(yva)); test_ds = SeqDataset(Xte, yte, np.ones_like(yte))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrfModel(d_feat=F, nhead=4, nlayers=2, d_ff=128, p=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none")

    best_val = float("inf"); best_state=None; wait=0
    for ep in range(1, EPOCHS+1):
        model.train(); total=0.0
        for xb, yb, wb in train_dl:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad(); logit = model(xb)
            loss = (criterion(logit, yb) * wb).mean()
            loss.backward(); opt.step()
            total += loss.item() * len(xb)
        model.eval(); vloss=0.0
        with torch.no_grad():
            for xb, yb, wb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logit = model(xb)
                loss = nn.functional.binary_cross_entropy_with_logits(logit, yb, reduction="mean")
                vloss += loss.item() * len(xb)
        vloss /= max(1, len(val_ds))
        print(f"[TRF] Epoch {ep:02d} | TrainLoss={total/max(1,len(train_ds)):.4f} | ValLoss={vloss:.4f}")
        if vloss < best_val - 1e-4:
            best_val = vloss; best_state = model.state_dict(); wait = 0
        else:
            wait += 1
            if wait >= PATIENCE: print(f"[TRF] Early stopping at {ep}"); break
    if best_state is not None: model.load_state_dict(best_state)

    def infer(dl):
        model.eval(); preds=[]
        with torch.no_grad():
            for xb, _, _ in dl:
                xb = xb.to(device)
                p = torch.sigmoid(model(xb)).cpu().numpy()
                preds.append(p)
        return np.concatenate(preds) if preds else np.array([])

    val_p  = infer(val_dl); test_p = infer(test_dl)

    best_thr, best_acc = 0.5, -1
    for thr in np.linspace(0.3,0.7,41):
        acc = accuracy_score(yva, (val_p>=thr).astype(int))
        if acc>best_acc: best_acc, best_thr=acc, thr
    val_auc  = roc_auc_score(yva, val_p) if len(np.unique(yva))>1 else float("nan")
    test_auc = roc_auc_score(yte, test_p) if len(np.unique(yte))>1 else float("nan")
    test_acc = accuracy_score(yte, (test_p>=best_thr).astype(int))
    print(f"[TRF]  Val AUC={val_auc:.3f} | thr={best_thr:.2f} (Acc={best_acc:.3f})")
    print(f"[TRF]  Test AUC={test_auc:.3f} | Acc(thr={best_thr:.2f})={test_acc:.3f}")

    pd.DataFrame({"p":val_p}).to_csv("trf_val_probs.csv", index=False)
    pd.DataFrame({"p":test_p}).to_csv("trf_test_probs.csv", index=False)

    return {"name":"trf","val_p":val_p,"test_p":test_p,"y_val":yva,"y_test":yte,"thr":best_thr,"val_auc":val_auc,"test_auc":test_auc}


# graph
def _maybe_show_or_save(name):
    if SHOW_FIGS: plt.show()
    else: plt.savefig(f"{name}.png", dpi=180); print(f"已保存: {name}.png"); plt.close()

def plot_roc_compare(models, set_name="val", name="roc"):
    plt.figure(figsize=(7,6))
    for o in models:
        y = o["y_val"] if set_name=="val" else o["y_test"]
        p = o["val_p"] if set_name=="val" else o["test_p"]
        if len(y)==0: continue
        try:
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            label = f"{o['name'].upper()} (AUC={auc:.3f})"
            plt.plot(fpr, tpr, lw=2, label=label)
        except Exception as e:
            print(f"[ROC] 跳过 {o['name']}：{repr(e)}")
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {set_name.upper()}")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    _maybe_show_or_save(f"{name}_{set_name}")

def plot_auc_bar(summary_rows, name="auc_bar"):
    df_cmp = pd.DataFrame(summary_rows, columns=["Model","Val AUC","Test AUC","Best Thr"])
    fig, ax = plt.subplots(1,1, figsize=(7,4))
    x = np.arange(len(df_cmp))
    ax.bar(x-0.2, df_cmp["Val AUC"], width=0.4, label="Val AUC")
    ax.bar(x+0.2, df_cmp["Test AUC"], width=0.4, label="Test AUC")
    ax.set_xticks(x); ax.set_xticklabels(df_cmp["Model"])
    ax.set_ylim(0.4, 0.7); ax.set_ylabel("AUC"); ax.set_title("AUC 对比")
    for i,v in enumerate(df_cmp["Test AUC"]):
        ax.text(i+0.22, v+0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.legend(); plt.tight_layout()
    _maybe_show_or_save(name)

def _plot_single_cm(ax, cm, title="", normalize=False):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if not normalize else 1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}", ha="center", va="center", color="black")
    return im

def plot_confusion_matrices(models, blend, name_counts="cm_counts", name_norm="cm_norm"):
    items = []
    for o in models:
        if len(o["y_test"])==0: continue
        y = o["y_test"]; p = o["test_p"]; thr = o["thr"]
        yhat = (p>=thr).astype(int)
        items.append((o["name"].upper(), y, yhat))
    if blend is not None:
        items.append(("BLEND", blend["y_test"], (blend["p_test"]>=blend["thr"]).astype(int)))

    n = len(items); cols = 2 if n<=2 else 2 if n==4 else 3
    rows = int(np.ceil(n/cols))

    # counts
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = np.array(axes).reshape(-1)
    for ax, (name, y, yhat) in zip(axes, items):
        cm = confusion_matrix(y, yhat, labels=[0,1])
        _plot_single_cm(ax, cm, title=f"{name} (counts)", normalize=False)
    for k in range(len(items), len(axes)): axes[k].axis("off")
    plt.tight_layout(); _maybe_show_or_save(name_counts)

    # normalized
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = np.array(axes).reshape(-1)
    for ax, (name, y, yhat) in zip(axes, items):
        cm = confusion_matrix(y, yhat, labels=[0,1]).astype(float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        _plot_single_cm(ax, cm_norm, title=f"{name} (normalized)", normalize=True)
    for k in range(len(items), len(axes)): axes[k].axis("off")
    plt.tight_layout(); _maybe_show_or_save(name_norm)

def plot_acc_curve(models, blend=None, set_name="val", name="acc_curve", thr_grid=None):
    if thr_grid is None: thr_grid = np.linspace(0.30, 0.70, 41)
    plt.figure(figsize=(7, 6))
    for o in models:
        y = o["y_val"] if set_name == "val" else o["y_test"]
        p = o["val_p"] if set_name == "val" else o["test_p"]
        if len(y) == 0: continue
        accs = [accuracy_score(y, (p >= t).astype(int)) for t in thr_grid]
        best_i = int(np.argmax(accs))
        label = f"{o['name'].upper()} (max={accs[best_i]:.3f}@{thr_grid[best_i]:.2f})"
        plt.plot(thr_grid, accs, lw=2, label=label)
    if blend is not None:
        yb = blend["y_val"] if set_name == "val" and "y_val" in blend else blend["y_test"]
        pb = blend["p_val"] if set_name == "val" and "p_val" in blend else blend["p_test"]
        accs = [accuracy_score(yb, (pb >= t).astype(int)) for t in thr_grid]
        best_i = int(np.argmax(accs))
        plt.plot(thr_grid, accs, lw=2, linestyle="--", label=f"BLEND (max={accs[best_i]:.3f}@{thr_grid[best_i]:.2f})")
    plt.xlabel("Decision threshold"); plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Threshold - {set_name.upper()}")
    plt.ylim(0.3, 1.0); plt.grid(alpha=0.3); plt.legend(loc="lower right")
    plt.tight_layout(); _maybe_show_or_save(f"{name}_{set_name}")


# PR, calibration, monthly AUC, AUC per ticket, straight side, equity, XGB
def plot_pr_curve(y, p, title="PR"):
    precision, recall, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, lw=2, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    _maybe_show_or_save(title)# PR

def plot_calibration(y, p, title="Calibration", n_bins=10):
    prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y, p)
    plt.figure(figsize=(6,5))
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.plot(prob_pred, prob_true, marker='o', lw=2, label=f"Brier={brier:.3f}")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(title); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    _maybe_show_or_save(title) #calibration

def plot_monthly_auc(y, p, dates, title="Monthly_AUC"):

    months = pd.to_datetime(dates).to_period('M')
    dfm = pd.DataFrame({'y': y, 'p': p, 'month': months})
    aucs=[]
    for m, g in dfm.groupby('month'):
        if g['y'].nunique()>1:
            aucs.append((m.to_timestamp(), roc_auc_score(g['y'], g['p'])))
    if not aucs:
        print(f"[{title}] No available months。")
        return
    aucs = pd.DataFrame(aucs, columns=["month","AUC"]).sort_values("month")
    plt.figure(figsize=(8,4))
    plt.plot(aucs["month"].astype(str), aucs["AUC"], marker='o')
    plt.xticks(rotation=45); plt.ylim(0.4,0.7); plt.grid(alpha=0.3)
    plt.title(title); plt.ylabel("AUC"); plt.tight_layout()
    _maybe_show_or_save(title)#monthly AUC

def per_ticker_auc(y, p, tick, title="PerTicker_AUC"):
    rows=[]
    for tk in np.unique(tick):
        m = (tick==tk)
        if len(np.unique(y[m]))>1:
            rows.append((tk, roc_auc_score(y[m], p[m])))
    if not rows: return
    auc_df = pd.DataFrame(rows, columns=["ticker","AUC"]).sort_values("AUC")
    plt.figure(figsize=(8, max(4,len(rows)//5)))
    plt.barh(auc_df["ticker"], auc_df["AUC"])
    plt.xlabel("AUC"); plt.title(title); plt.xlim(0.4,0.8)
    plt.tight_layout(); _maybe_show_or_save(title)#AUC per ticket

def plot_score_hist(y, p, title="Score_Hist"):
    plt.figure(figsize=(6,4))
    plt.hist(p[y==0], bins=30, alpha=0.6, density=True, label="Down", color="#1f77b4")
    plt.hist(p[y==1], bins=30, alpha=0.6, density=True, label="Up",   color="#ff7f0e")
    plt.xlabel("Predicted probability"); plt.ylabel("Density"); plt.title(title)
    plt.legend(); plt.tight_layout(); _maybe_show_or_save(title) #straight side

def plot_equity_longshort(df_test, scores, title="Equity_LS"):

    if len(df_test) != len(scores):
        n = min(len(df_test), len(scores))
        df_test = df_test.iloc[:n].copy()
        scores = np.asarray(scores)[:n]
        print(f"[{title}] The lengths are inconsistent and have been automatically aligned {n} 行（df={len(df_test)}, scores={len(scores)}）")

    sub = df_test[["date","ticker","y_reg_h"]].copy()
    sub["score"] = scores
    sub["rank"] = sub.groupby("date")["score"].rank(pct=True)
    long = sub[sub["rank"]>=0.8].groupby("date")["y_reg_h"].mean()
    short= sub[sub["rank"]<=0.2].groupby("date")["y_reg_h"].mean()
    ls = (long - short).reindex(sorted(sub["date"].unique())).fillna(0.0)
    eq = (1.0 + ls).cumprod()
    plt.figure(figsize=(8,4)); plt.plot(eq.index, eq.values)
    plt.title("Long-Short Equity Curve (Top20% - Bottom20%)"); plt.ylabel("Cumulative factor")
    plt.grid(alpha=0.3); plt.tight_layout(); _maybe_show_or_save(title)#equity

def plot_xgb_importance(booster, topn=20, title="xgb_importance"):
    fmap = booster.get_score(importance_type="gain")
    if not fmap: return
    imp = pd.Series(fmap).sort_values(ascending=False).head(topn)
    plt.figure(figsize=(6,5)); imp[::-1].plot(kind="barh")
    plt.title("XGB Importance (gain) Top-20"); plt.tight_layout(); _maybe_show_or_save(title) #equity

# PR-AUC under class imbalance & Ensemble gains
def plot_pr_trf_vs_xgb(out_trf, out_xgb, name="Fig_6_2_PR_Test"):

    # 保护：任一模型无测试集则跳过
    if len(out_trf.get("y_test", [])) == 0 or len(out_xgb.get("y_test", [])) == 0:
        print(f"[{name}] Test set predictions lacking TRF or XGB have been skipped")
        return

    y_trf, p_trf = out_trf["y_test"], out_trf["test_p"]
    y_xgb, p_xgb = out_xgb["y_test"], out_xgb["test_p"]


    y_all = np.concatenate([y_trf, y_xgb])
    pos_rate = float(np.mean(y_all)) if len(y_all) else 0.0

    pr_trf = precision_recall_curve(y_trf, p_trf)
    pr_xgb = precision_recall_curve(y_xgb, p_xgb)
    ap_trf = average_precision_score(y_trf, p_trf)
    ap_xgb = average_precision_score(y_xgb, p_xgb)

    plt.figure(figsize=(6.2, 5.2))

    plt.plot(pr_trf[1], pr_trf[0], lw=2.2, label=f"TRF (AP={ap_trf:.3f})")
    plt.plot(pr_xgb[1], pr_xgb[0], lw=2.0, linestyle="--", label=f"XGB (AP={ap_xgb:.3f})")
    plt.hlines(pos_rate, 0, 1, linestyles=":", linewidth=1.5, label=f"No-skill={pos_rate:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR — Test (Class Imbalance)")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    _maybe_show_or_save(name)


def plot_ensemble_vs_best_single(models, blend, name="Fig_6_9_Ensemble_vs_Best"):

    cands = []
    for o in models:
        if len(o.get("y_test", [])) == 0 or not np.isfinite(o.get("test_auc", np.nan)):
            continue
        y, p, thr = o["y_test"], o["test_p"], o["thr"]
        acc = accuracy_score(y, (p >= thr).astype(int))
        cands.append((o["name"].upper(), float(o["test_auc"]), float(acc)))

    if not cands or blend is None or len(blend.get("y_test", [])) == 0:
        print(f"[{name}] There are no available candidates or fusion results")
        return

    # Select the single model with the best Test AUC
    best_name, best_auc, best_acc = max(cands, key=lambda t: t[1])

    #Fusion metrics (calculated directly using the passed-in blend object to avoid external dependencies)
    yb, pb, thrb = blend["y_test"], blend["p_test"], blend["thr"]
    blend_auc = roc_auc_score(yb, pb) if len(np.unique(yb)) > 1 else float("nan")
    blend_acc = accuracy_score(yb, (pb >= thrb).astype(int))

    # Print gain/deficiency summary
    d_auc = blend_auc - best_auc
    d_acc = blend_acc - best_acc
    sign_auc = "↑" if d_auc >= 0 else "↓"
    sign_acc = "↑" if d_acc >= 0 else "↓"
    print("\n[Ensemble gains & limitations]")
    print(f"Best single: {best_name} | Test AUC={best_auc:.3f} | Acc={best_acc:.3f}")
    print(f"Ensemble   : BLEND     | Test AUC={blend_auc:.3f} | Acc={blend_acc:.3f}")
    print(f"ΔAUC={d_auc:+.3f} {sign_auc} | ΔAcc={d_acc:+.3f} {sign_acc}")

    # Bar chart
    metrics = ["AUC", "Accuracy"]
    best_vals = [best_auc, best_acc]
    blend_vals = [blend_auc, blend_acc]

    x = np.arange(len(metrics))
    w = 0.36
    plt.figure(figsize=(6.8, 4.6))
    plt.bar(x - w/2, best_vals, width=w, label=f"Best single ({best_name})")
    plt.bar(x + w/2, blend_vals, width=w, label="Ensemble (BLEND)")
    for xi, v in zip(x - w/2, best_vals):
        plt.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + w/2, blend_vals):
        plt.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, metrics)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Ensemble vs Best Single (Test)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="upper left")
    plt.tight_layout()
    _maybe_show_or_save(name)

# Calibration & Brier reliability
def plot_calibration_side_by_side(y_xgb, p_xgb, y_blend, p_blend, n_bins=10, name="Fig_6_5_6_Calibration_Test"):


    if len(y_xgb) == 0 or len(y_blend) == 0:
        print(f"[{name}] 缺少测试集数据，已跳过。")
        return

    prob_true_x, prob_pred_x = calibration_curve(y_xgb, p_xgb, n_bins=n_bins, strategy="quantile")
    brier_x = brier_score_loss(y_xgb, p_xgb)

    prob_true_b, prob_pred_b = calibration_curve(y_blend, p_blend, n_bins=n_bins, strategy="quantile")
    brier_b = brier_score_loss(y_blend, p_blend)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)

    # XGB
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(prob_pred_x, prob_true_x, marker="o", lw=2, label=f"Brier={brier_x:.3f}")
    ax.set_title("Fig. 6-5: Calibration—XGB (Test)", fontsize=10)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(loc="upper left")

    # Ensemble
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(prob_pred_b, prob_true_b, marker="o", lw=2, label=f"Brier={brier_b:.3f}")
    ax.set_title("Fig. 6-6: Calibration—Ensemble (Test)", fontsize=10)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(loc="upper left")

    fig.suptitle("Reliability Diagrams (Quantile-binned)", fontsize=11)
    plt.tight_layout()
    _maybe_show_or_save(name)




# fusion
def blend_three(y_val, p1, p2, p3, step=0.05):
    best_auc, best_w = -1, (1/3,1/3,1/3)
    grid = np.arange(0,1+1e-9, step)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9 or w3>1: continue
            p = w1*p1 + w2*p2 + w3*p3
            auc = roc_auc_score(y_val, p)
            if auc > best_auc:
                best_auc, best_w = auc, (w1,w2,w3)
    return best_auc, best_w


#  main function
def main():
    df, base_cols, is_train, is_val, is_test = load_and_featurize()

    pd.DataFrame({"y": df.loc[is_val,"y_cls"].values}).to_csv("y_val.csv", index=False)
    pd.DataFrame({"y": df.loc[is_test,"y_cls"].values}).to_csv("y_test.csv", index=False)

    print("\n=== 训练 XGBoost ===")
    out_xgb = train_xgb(df, base_cols, is_train, is_val, is_test)

    print("\n=== 训练 Ridge Logistic ===")
    out_ridge = train_ridge(df, base_cols, is_train, is_val, is_test)

    print("\n=== 训练 Transformer ===")
    out_trf = train_transformer(df, base_cols, is_train, is_val, is_test)

    # model comparison
    rows = []
    for o in [out_xgb, out_ridge, out_trf]:
        rows.append([o["name"].upper(), o["val_auc"], o["test_auc"], o["thr"]])
    print("\n=== 单模型对比（Val/Test AUC，最佳阈值） ===")
    print(pd.DataFrame(rows, columns=["Model","Val AUC","Test AUC","Best Thr"]).to_string(index=False))

    # Basic diagram
    plot_roc_compare([out_xgb, out_ridge, out_trf], set_name="val",  name="roc")
    plot_roc_compare([out_xgb, out_ridge, out_trf], set_name="test", name="roc")
    plot_auc_bar(rows, name="auc_bar")

    # fusion
    n_min_val = min(len(out_xgb["val_p"]), len(out_ridge["val_p"]), len(out_trf["val_p"]))
    n_min_tst = min(len(out_xgb["test_p"]),len(out_ridge["test_p"]),len(out_trf["test_p"]))
    pv_x, pv_r, pv_t = out_xgb["val_p"][:n_min_val], out_ridge["val_p"][:n_min_val], out_trf["val_p"][:n_min_val]
    pt_x, pt_r, pt_t = out_xgb["test_p"][:n_min_tst],out_ridge["test_p"][:n_min_tst],out_trf["test_p"][:n_min_tst]
    yv = out_xgb["y_val"][:n_min_val]; yt = out_xgb["y_test"][:n_min_tst]

    val_auc_blend, (wx, wr, wt) = blend_three(yv, pv_x, pv_r, pv_t, step=0.05)
    p_val_blend  = wx*pv_x + wr*pv_r + wt*pv_t
    p_test_blend = wx*pt_x + wr*pt_r + wt*pt_t
    test_auc_blend = roc_auc_score(yt, p_test_blend) if len(np.unique(yt))>1 else float("nan")
    best_thr, best_acc = 0.5, -1
    for thr in np.linspace(0.3,0.7,41):
        acc = accuracy_score(yv, (p_val_blend>=thr).astype(int))
        if acc>best_acc: best_acc, best_thr=acc, thr
    test_acc_blend = accuracy_score(yt, (p_test_blend>=best_thr).astype(int))
    print("\nWeighted fusion of three models")
    print(f"权重 w=(XGB {wx:.2f}, RIDGE {wr:.2f}, TRF {wt:.2f}) | Val AUC={val_auc_blend:.3f}")
    print(f"Blend Test AUC={test_auc_blend:.3f} | Acc(thr={best_thr:.2f})={test_acc_blend:.3f}")
    pd.DataFrame({"p": p_test_blend}).to_csv("blend_test_probs.csv", index=False)

    blend_obj_full = {"y_val": yv, "p_val": p_val_blend, "y_test": yt, "p_test": p_test_blend, "thr": best_thr}
    plot_acc_curve([out_xgb, out_ridge, out_trf], blend_obj_full, set_name="val",  name="acc_curve")
    plot_acc_curve([out_xgb, out_ridge, out_trf], blend_obj_full, set_name="test", name="acc_curve")

    blend_obj = {"y_test": yt, "p_test": p_test_blend, "thr": best_thr}
    plot_confusion_matrices([out_xgb, out_ridge, out_trf], blend_obj,
                            name_counts="cm_test_counts", name_norm="cm_test_norm")


    # PR （Test）
    plot_pr_curve(out_xgb["y_test"],   out_xgb["test_p"],   "PR_XGB_TEST")
    plot_pr_curve(out_ridge["y_test"], out_ridge["test_p"], "PR_RIDGE_TEST")
    plot_pr_curve(out_trf["y_test"],   out_trf["test_p"],   "PR_TRF_TEST")
    plot_pr_curve(yt,                   p_test_blend,        "PR_BLEND_TEST")

    # Calibration curve（Test）
    plot_calibration(out_xgb["y_test"], out_xgb["test_p"], "Calibration_XGB_TEST")
    plot_calibration(yt, p_test_blend,                         "Calibration_BLEND_TEST")

    # Monthly AUC
    df_test = df.loc[is_test].copy()
    dates_test   = df_test["date"].values
    tickers_test = df_test["ticker"].values
    plot_monthly_auc(out_xgb["y_test"], out_xgb["test_p"], dates_test, "Monthly_AUC_XGB_TEST")

    # AUC (XGB) of each stock
    per_ticker_auc(out_xgb["y_test"], out_xgb["test_p"], tickers_test, "PerTicker_AUC_XGB_TEST")

    # Fraction Blend
    plot_score_hist(yt, p_test_blend, "Score_Hist_BLEND_TEST")

    #L-S equity curve
    plot_equity_longshort(df_test, out_xgb["test_p"], "Equity_LS_XGB_TEST")
    plot_equity_longshort(df_test, p_test_blend,      "Equity_LS_BLEND_TEST")

    # XGB Feature importance
    plot_xgb_importance(out_xgb["booster"], topn=20, title="xgb_importance")

    # TRF vs XGB
    plot_pr_trf_vs_xgb(out_trf, out_xgb, name="Fig_6_2_PR_Test")

    #  Ensemble vs Best single
    plot_ensemble_vs_best_single([out_xgb, out_ridge, out_trf], blend_obj, name="Fig_6_9_Ensemble_vs_Best")

    # （XGB）&（Ensemble）
    plot_calibration_side_by_side(
        out_xgb["y_test"], out_xgb["test_p"],
        yt, p_test_blend,
        n_bins=10,
        name="Fig_6_5_6_Calibration_Test"
    )


    print("\nAll done. The graphic has been displayed/saved. Output：xgb_*.csv / ridge_*.csv / trf_*.csv / blend_test_probs.csv / y_*.csv")
    return out_xgb, out_ridge, out_trf, yt, p_test_blend


# entrance
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="只保存图片，不弹出显示")
    args = parser.parse_args()
    SHOW_FIGS = not args.save
    main()
