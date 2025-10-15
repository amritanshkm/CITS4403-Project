# project-root/src/data_prep.py
from __future__ import annotations
import os, re, glob, shutil
from pathlib import Path
import pandas as pd
import numpy as np

def dpath(*parts) -> Path:
    root = os.getenv("PROJECT_ROOT", "")
    base = Path(root) if root else Path.cwd()
    return base.joinpath(*parts)

_QRE = re.compile(r"^(\d{4})-?Q([1-4])$", re.I)
def to_quarter(x):
    if pd.isna(x): return pd.NaT
    s = str(x).strip().upper().replace("–","-").replace("_","")
    m = _QRE.match(s)
    if m:
        return pd.Period(int(m.group(1)), int(m.group(2)), freq="Q-DEC")
    for fmt in (None,"%b-%y","%b-%Y","%Y-%m-%d","%Y-%m","%d/%m/%Y","%d-%m-%Y","%b %Y","%Y %b"):
        dt = pd.to_datetime(s, format=fmt, errors="coerce") if fmt else pd.to_datetime(s, errors="coerce")
        if not pd.isna(dt): return dt.to_period("Q-DEC")
    return pd.NaT

def norm_quarter_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out["Quarter"] = out[col].apply(to_quarter)
    return out.dropna(subset=["Quarter"])

def _first_existing(paths):
    for p in paths:
        if p and Path(p).exists(): return Path(p)
    return None

def _search_raw(name_frags):
    raw = dpath("data","raw")
    if not raw.exists(): return None
    fr = [f.lower() for f in name_frags]
    for p in sorted(raw.glob("*.csv")):
        n = p.name.lower().replace(" ","_")
        if all(f in n for f in fr): return p
    return None

def _read_with_header_fix(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, dtype=str)
    if df.empty: return df
    unnamed = sum(str(c).lower().startswith("unnamed") or str(c).strip()=="" for c in df.columns)
    first = df.iloc[0].astype(str).fillna("")
    if unnamed >= len(df.columns)//2 and (first.str.len()>0).sum() >= len(df.columns)//2:
        df.columns = [x.strip() or f"col_{i}" for i,x in enumerate(first.tolist())]
        df = df.iloc[1:].reset_index(drop=True)
    return df

def _pick_date_col(cols):
    cols = list(cols)
    for k in ["date","month","time","period","reference"]:
        for c in cols:
            if k in str(c).lower(): return c
    return cols[0] if len(cols)>0 else None

def _pick_rate_col(cols):
    cols = list(cols)
    for c in cols:
        cl = str(c).lower()
        if "seasonally" in cl and "adjusted" in cl and ("unemployment" in cl or "rate" in cl): return c
    for c in cols:
        cl = str(c).lower()
        if "unemployment" in cl and "rate" in cl: return c
    for c in cols:
        cl = str(c).lower()
        if "unemployment" in cl: return c
    for c in cols:
        cl = str(c).lower()
        if "rate" in cl: return c
    return cols[-1] if len(cols)>0 else None

def _parse_date_any(s):
    if pd.isna(s): return pd.NaT
    t = str(s).strip()
    if not t: return pd.NaT
    if _QRE.match(t):
        try: return pd.Period(t.replace("-",""), freq="Q-DEC").to_timestamp(how="start")
        except: pass
    for f in ["%b-%y","%b-%Y","%Y-%m","%Y/%m","%d/%m/%Y","%Y-%m-%d","%d-%m-%Y","%b %Y","%Y %b"]:
        try: return pd.to_datetime(t, format=f, errors="raise")
        except: pass
    if t.isdigit() and 5<=len(t)<=6:
        try: return pd.to_datetime(int(t), unit="D", origin="1899-12-30")
        except: pass
    return pd.to_datetime(t, errors="coerce")

def _build_unemp_from_monthly(raw_fp: Path) -> pd.DataFrame:
    df = _read_with_header_fix(raw_fp)
    if df.empty: return pd.DataFrame(columns=["Quarter","UnemploymentRate"])
    date_col = _pick_date_col(df.columns)
    rate_col = _pick_rate_col(df.columns)
    if date_col is None or rate_col is None:
        raise ValueError(f"Cannot find date/rate columns in {raw_fp.name}. Columns: {list(df.columns)}")

    t = df[[date_col, rate_col]].copy()
    t.columns = ["DateRaw","RateRaw"]
    t["Rate"] = (t["RateRaw"].astype(str).str.replace("%","",regex=False).str.replace(",","",regex=False).str.strip())
    t["Rate"] = pd.to_numeric(t["Rate"], errors="coerce")
    t["Date"] = t["DateRaw"].apply(_parse_date_any)
    t = t.dropna(subset=["Date","Rate"]).reset_index(drop=True)
    if t.empty: return pd.DataFrame(columns=["Quarter","UnemploymentRate"])

    t["Quarter"] = t["Date"].dt.to_period("Q-DEC")
    q = (t.groupby("Quarter", as_index=False)["Rate"].mean().rename(columns={"Rate":"UnemploymentRate"}))
    # keep a debug file
    out_dir = dpath("data","processed"); out_dir.mkdir(parents=True, exist_ok=True)
    t[["Date","Quarter","Rate"]].to_csv(out_dir/"unemployment_monthly_clean.csv", index=False)
    return q[["Quarter","UnemploymentRate"]].sort_values("Quarter").reset_index(drop=True)

def load_unemployment() -> pd.DataFrame:
    proc = dpath("data","processed","unemployment_quarterly.csv")
    if proc.exists():
        q = pd.read_csv(proc)
        if not q.empty and {"Quarter","UnemploymentRate"} <= set(map(str,q.columns)):
            q = norm_quarter_col(q, "Quarter")[["Quarter","UnemploymentRate"]].sort_values("Quarter").reset_index(drop=True)
            if not q.empty:
                print(f"[data_prep] Using existing unemployment_quarterly.csv | rows={len(q)}")
                return q
        print("[data_prep] Existing unemployment_quarterly.csv empty/invalid. Rebuilding from raw...")

    raw_candidates = [
        dpath("data","raw","Unemployment rate.csv"),
        _search_raw(["unemployment","rate"]),
        _search_raw(["table","3.1"]),
        _search_raw(["seasonally","adjusted"]),
    ]
    raw = _first_existing(raw_candidates)
    if raw is None:
        raise FileNotFoundError("Place monthly unemployment CSV in data/raw/ (e.g., 'Unemployment rate.csv').")
    q = _build_unemp_from_monthly(raw)
    out_dir = dpath("data","processed"); out_dir.mkdir(parents=True, exist_ok=True)
    q.to_csv(out_dir/"unemployment_quarterly.csv", index=False)
    print(f"[data_prep] Built unemployment_quarterly from '{raw.name}' | rows={len(q)}")
    return q

def load_insolvencies() -> pd.DataFrame:
    fp = _first_existing([
        dpath("data","raw","regional_quarterly_time_series.csv"),
        dpath("data","raw","quarterly_personal_insolvencies.csv"),
        _search_raw(["insolvenc","quarter"]),
        _search_raw(["bankrupt","quarter"]),
    ])
    if fp is None:
        raise FileNotFoundError("AFSA CSV not found in data/raw/ (regional_quarterly_time_series.csv or quarterly_personal_insolvencies.csv).")
    df = pd.read_csv(fp, low_memory=False)
    if df.empty: return pd.DataFrame(columns=["Quarter","Insolvencies"])

    qcol = next((c for c in df.columns if "quarter" in str(c).lower()), df.columns[0])
    vcol = next((c for c in df.columns if any(k in str(c).lower() for k in ["number of people","insolvenc","bankrupt","total"])), df.columns[-1])

    out = df[[qcol, vcol]].copy()
    out.columns = ["Quarter","Insolvencies"]
    out["Quarter"] = out["Quarter"].apply(to_quarter)
    out["Insolvencies"] = pd.to_numeric(out["Insolvencies"], errors="coerce")
    out = out.dropna(subset=["Quarter"])
    out = out.groupby("Quarter", as_index=False)["Insolvencies"].sum().sort_values("Quarter").reset_index(drop=True)
    print(f"[data_prep] Insolvencies source: {fp.name} | rows={len(out)}")
    dpath("data","processed").mkdir(parents=True, exist_ok=True)
    out.to_csv(dpath("data","processed","afsa_insolvencies_quarterly.csv"), index=False)
    return out

def _to_str_quarter(p: pd.Period) -> str:
    return f"{p.year}Q{p.quarter}"

def make_merges(unemp, afsa):
    out_dir = dpath("data","processed"); out_dir.mkdir(parents=True, exist_ok=True)
    merged_left  = afsa.merge(unemp, on="Quarter", how="left").sort_values("Quarter").reset_index(drop=True)
    merged_inner = afsa.merge(unemp, on="Quarter", how="inner").sort_values("Quarter").reset_index(drop=True)
    merged_left.to_csv(out_dir/"merged_macro.csv", index=False)
    merged_inner.to_csv(out_dir/"merged_overlap.csv", index=False)

    if not unemp.empty:
        q_min, q_max = unemp["Quarter"].min(), unemp["Quarter"].max()
        q_index = pd.period_range(q_min, q_max, freq="Q-DEC")
        u_full = (unemp.set_index("Quarter").reindex(q_index).rename_axis("Quarter").reset_index())
        u_full.columns = ["Quarter","UnemploymentRate"]
        u_full["UnemploymentRate"] = u_full["UnemploymentRate"].interpolate(limit_direction="both")
        merged_filled = afsa.merge(u_full, on="Quarter", how="left").sort_values("Quarter").reset_index(drop=True)
    else:
        merged_filled = merged_left.copy()
    merged_filled.to_csv(out_dir/"merged_filled.csv", index=False)
    return merged_left, merged_inner, merged_filled

def build_final_model(merged_inner):
    out_dir = dpath("data","processed")
    df = merged_inner.copy().sort_values("Quarter").reset_index(drop=True)
    if df.empty:
        df = pd.read_csv(out_dir/"merged_macro.csv")
        df = norm_quarter_col(df, "Quarter")[["Quarter","Insolvencies","UnemploymentRate"]].sort_values("Quarter").reset_index(drop=True)

    df["Insolvencies_lag1"] = df["Insolvencies"].shift(1)
    df["Insolvencies_lag2"] = df["Insolvencies"].shift(2)
    df["Insolvencies_lag4"] = df["Insolvencies"].shift(4)
    df["UnemploymentRate_lag1"] = df["UnemploymentRate"].shift(1)
    df["Insolvencies_nextQ"] = df["Insolvencies"].shift(-1)

    keep = df.dropna(subset=[
        "Insolvencies","UnemploymentRate",
        "Insolvencies_lag1","Insolvencies_lag2","Insolvencies_lag4",
        "UnemploymentRate_lag1","Insolvencies_nextQ"
    ]).copy()

    if isinstance(keep["Quarter"].iloc[0], pd.Period):
        keep["Quarter"] = keep["Quarter"].apply(_to_str_quarter)
    else:
        keep["Quarter"] = keep["Quarter"].apply(lambda x: _to_str_quarter(to_quarter(x)))

    keep.to_csv(out_dir/"final_model.csv", index=False)
    print(f"[data_prep] Saved final_model.csv | rows={len(keep)} | path={out_dir/'final_model.csv'}")
    return keep

def main():
    # ensure dirs
    dpath("data","raw").mkdir(parents=True, exist_ok=True)
    dpath("data","processed").mkdir(parents=True, exist_ok=True)

    unemp = load_unemployment()
    afsa  = load_insolvencies()
    # save tidy inputs
    unemp.to_csv(dpath("data","processed","unemployment_quarterly.csv"), index=False)
    afsa.to_csv(dpath("data","processed","afsa_insolvencies_quarterly.csv"), index=False)

    merged_left, merged_inner, merged_filled = make_merges(unemp, afsa)
    model = build_final_model(merged_inner)

    nn = lambda s: int(pd.Series(s).notna().sum())
    print(f"[data_prep] Unemployment rows: {len(unemp)} | AFSA rows: {len(afsa)}")
    print(f"[data_prep] LEFT   -> rows={len(merged_left)}  | Unemp non-null={nn(merged_left['UnemploymentRate'])}/{len(merged_left)}  (merged_macro.csv)")
    print(f"[data_prep] INNER  -> rows={len(merged_inner)} | Unemp non-null={nn(merged_inner['UnemploymentRate'])}/{len(merged_inner)} (merged_overlap.csv)")
    print(f"[data_prep] FILLED -> rows={len(merged_filled)} | Unemp non-null={nn(merged_filled['UnemploymentRate'])}/{len(merged_filled)} (merged_filled.csv)")
    print(f"[data_prep] FINAL  -> rows={len(model)} (final_model.csv)")

if __name__ == "__main__":
    main()
