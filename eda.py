"""
EDA: быстрый анализ данных хакатона (отгрузки со складов).
Запуск: python eda.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Загрузка ──────────────────────────────────────────────
train = pd.read_parquet("train_solo_track.parquet")
test  = pd.read_parquet("test_solo_track.parquet")

print("=" * 60)
print("TRAIN")
print(f"  shape : {train.shape}")
print(f"  dtypes:\n{train.dtypes.to_string()}")
print(f"  nulls :\n{train.isnull().sum().to_string()}")
print(f"  time  : {train['timestamp'].min()} -- {train['timestamp'].max()}")
print(f"  routes: {train['route_id'].nunique()}")

print("\nTEST")
print(f"  shape : {test.shape}")
print(f"  dtypes:\n{test.dtypes.to_string()}")
print(f"  nulls :\n{test.isnull().sum().to_string()}")
print(f"  time  : {test['timestamp'].min()} -- {test['timestamp'].max()}")
print(f"  routes: {test['route_id'].nunique()}")

# ── Target статистики ─────────────────────────────────────
t = train["target_1h"]
print("\n" + "=" * 60)
print("TARGET_1H statistics")
print(t.describe())
print(f"  zeros   : {(t == 0).sum()} ({(t == 0).mean():.2%})")
print(f"  p99     : {t.quantile(0.99):.2f}")
print(f"  p999    : {t.quantile(0.999):.2f}")
print(f"  skewness: {t.skew():.2f}")

# ── Гистограмма target ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(t, bins=100, edgecolor="k", alpha=0.7)
axes[0].set_title("target_1h distribution")
axes[0].set_xlabel("target_1h")
axes[0].set_ylabel("count")

axes[1].hist(t[t > 0], bins=100, edgecolor="k", alpha=0.7)
axes[1].set_title("target_1h distribution (>0)")
axes[1].set_xlabel("target_1h")
axes[1].set_ylabel("count")
plt.tight_layout()
plt.savefig("eda_target_hist.png", dpi=150)
print("\nСохранено: eda_target_hist.png")

# ── Временной ряд нескольких маршрутов ────────────────────
sample_routes = train["route_id"].unique()[:5]
fig, axes = plt.subplots(len(sample_routes), 1, figsize=(16, 3 * len(sample_routes)), sharex=True)
for i, rid in enumerate(sample_routes):
    sub = train[train["route_id"] == rid].sort_values("timestamp")
    axes[i].plot(sub["timestamp"], sub["target_1h"], linewidth=0.5)
    axes[i].set_title(f"route_id={rid}")
    axes[i].set_ylabel("target_1h")
axes[-1].set_xlabel("timestamp")
plt.tight_layout()
plt.savefig("eda_routes_ts.png", dpi=150)
print("Сохранено: eda_routes_ts.png")

# ── Горизонт теста ────────────────────────────────────────
print("\n" + "=" * 60)
print("ГОРИЗОНТ ТЕСТА")
last_train_ts = train.groupby("route_id")["timestamp"].max()
test_merged = test.merge(last_train_ts.rename("last_train_ts"), on="route_id")
test_merged["horizon_steps"] = (
    (test_merged["timestamp"] - test_merged["last_train_ts"]).dt.total_seconds() / 1800
).astype(int)
print(test_merged["horizon_steps"].value_counts().sort_index())
print(f"\nМакс горизонт: {test_merged['horizon_steps'].max()} шагов")
print(f"Мин горизонт : {test_merged['horizon_steps'].min()} шагов")

# ── Средний профиль по часам ──────────────────────────────
train["hour"] = train["timestamp"].dt.hour
hourly = train.groupby("hour")["target_1h"].mean()
fig, ax = plt.subplots(figsize=(10, 4))
hourly.plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Средний target_1h по часам")
ax.set_xlabel("hour")
ax.set_ylabel("mean target_1h")
plt.tight_layout()
plt.savefig("eda_hourly_profile.png", dpi=150)
print("Сохранено: eda_hourly_profile.png")

# ── Status корреляции ─────────────────────────────────────
status_cols = [c for c in train.columns if c.startswith("status_")]
if status_cols:
    corr = train[status_cols + ["target_1h"]].corr()["target_1h"].drop("target_1h")
    print("\n" + "=" * 60)
    print("Корреляции status → target_1h:")
    print(corr.to_string())

print("\n✓ EDA завершён")
