"""
Weighted Efficiency Score v1
===========================================================
Computes Weighted Efficiency (WE) on trade execution list.

Run directly to:
  (a) validate the spec's worked example (WE ≈ 60.3800), and
  (b) score a CSV passed on the command line:
        python we_v1.py my_trades.csv
  Required CSV columns : pnl, exit_date
  Optional CSV columns : exit_time, gross_pnl, charges

All constants are fixed anchors — DO NOT tune to any strategy.
Changing them requires a version bump of the spec.
"""

from __future__ import annotations

import csv
import math
from datetime import date, datetime
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Constants (Section 10 / 15) — DO NOT TUNE
# ---------------------------------------------------------------------------
EXPECTANCY_ANCHOR_R = 0.30
RR_ANCHOR           = 1.50
STREAK_DECAY        = 3.0
RETURN_ANCHOR_INR   = 25_000.0
STABILITY_MONTHLY_W = 0.50

WEIGHT_EXPECTANCY = 0.30
WEIGHT_RR         = 0.20
WEIGHT_STREAK     = 0.20
WEIGHT_RETURN     = 0.15
WEIGHT_STABILITY  = 0.15

assert abs(WEIGHT_EXPECTANCY + WEIGHT_RR + WEIGHT_STREAK
           + WEIGHT_RETURN + WEIGHT_STABILITY - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pstdev(xs: Sequence[float]) -> float:
    """Population standard deviation (as required by §5b)."""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _iso_week_key(d: date) -> tuple[int, int]:
    iso = d.isocalendar()
    return (iso[0], iso[1])


def _month_key(d: date) -> tuple[int, int]:
    return (d.year, d.month)


def _bucket_score(bucket_sums: Sequence[float]) -> float:
    """§5c: 100 * Positive_Fraction * 1/(1+CV), in [0, 100]."""
    if not bucket_sums:
        return 0.0
    n = len(bucket_sums)
    n_pos = sum(1 for s in bucket_sums if s > 0)
    pos_frac = n_pos / n
    mean_b = _mean(bucket_sums)
    if n < 2 or mean_b == 0:
        cv = 1.0
    else:
        cv = _pstdev(bucket_sums) / abs(mean_b)
    return 100.0 * pos_frac * (1.0 / (1.0 + cv))


def _parse_date(v) -> date:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    s = str(v).strip()
    # tolerate "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS"
    return datetime.fromisoformat(s.replace(" ", "T")).date()


# ---------------------------------------------------------------------------
# Trade normalization
# ---------------------------------------------------------------------------
def _normalize_trades(trades: Iterable[dict]) -> list[dict]:
    """
    Accepts dicts with keys: pnl, exit_date, [exit_time], [gross_pnl], [charges].
    Returns a list sorted by (exit_date, exit_time) — required by §1 and
    anti-pattern #7.
    """
    out = []
    for t in trades:
        ed = _parse_date(t["exit_date"])

        et = t.get("exit_time", "00:00:00")
        if isinstance(et, str):
            et = et.strip() or "00:00:00"
        elif hasattr(et, "isoformat"):
            et = et.isoformat()
        else:
            et = str(et)

        out.append({
            "pnl":       float(t["pnl"]),
            "exit_date": ed,
            "exit_time": et,
            "gross_pnl": float(t.get("gross_pnl", t["pnl"])),
            "charges":   float(t.get("charges", 0.0)),
        })
    out.sort(key=lambda r: (r["exit_date"], r["exit_time"]))
    return out


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_we(
    trades: Iterable[dict],
    *,
    label: str = "All",
    start_capital: float = 1_000_000.0,
) -> dict:
    """
    Compute WE and all §8 output columns for ONE trade list.

    Parameters
    ----------
    trades : iterable of dict
        Required keys: ``pnl`` (float), ``exit_date`` (date / ISO string).
        Optional keys: ``exit_time`` (used for chronological tie-breaking),
        ``gross_pnl``, ``charges`` (reporting only).
    label : str
        Free-text label for the "Dataset" column (default "All").
    start_capital : float
        Reference invested capital for drawdown & % return reporting.

    Returns
    -------
    dict with all §8 output columns (plus a few ``_``-prefixed diagnostics).
    """
    rows = _normalize_trades(trades)
    N = len(rows)
    pnl = [r["pnl"] for r in rows]

    # ---- Step 1: raw metrics ------------------------------------------------
    W = sum(1 for p in pnl if p > 0)
    L = sum(1 for p in pnl if p < 0)

    Net_PnL       = sum(pnl)
    Gross_PnL     = sum(r["gross_pnl"] for r in rows)
    Total_Charges = sum(r["charges"]   for r in rows)
    Win_Rate      = (W / N) if N > 0 else 0.0

    # ---- Step 2: avg win / avg loss / RR ------------------------------------
    wins   = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p < 0]
    Avg_Win  = _mean(wins)   if W > 0 else 0.0
    Avg_Loss = abs(_mean(losses)) if L > 0 else 0.0
    RR       = (Avg_Win / Avg_Loss) if Avg_Loss > 0 else float("nan")

    # ---- Step 3: expectancy in R units --------------------------------------
    if math.isnan(RR):
        Expectancy_R  = 0.0
        rr_edge_case  = "no losing trades"
    else:
        Expectancy_R  = Win_Rate * RR - (1.0 - Win_Rate)
        rr_edge_case  = None

    # ---- Step 4: losing-streak statistics -----------------------------------
    runs, cur = [], 0
    for p in pnl:
        if p < 0:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)

    Max_Streak    = max(runs) if runs else 0
    Avg_Streak    = _mean(runs) if runs else 0.0
    Excess_Streak = max(0.0, Max_Streak - Avg_Streak)

    # ---- Step 5: monthly / weekly stability ---------------------------------
    monthly, weekly = {}, {}
    for r in rows:
        mk = _month_key(r["exit_date"])
        wk = _iso_week_key(r["exit_date"])
        monthly[mk] = monthly.get(mk, 0.0) + r["pnl"]
        weekly[wk]  = weekly.get(wk,  0.0) + r["pnl"]

    monthly_buckets = list(monthly.values())
    weekly_buckets  = list(weekly.values())

    Monthly_Score   = _bucket_score(monthly_buckets)
    Weekly_Score    = _bucket_score(weekly_buckets)
    Stability_Score = STABILITY_MONTHLY_W * Monthly_Score \
                    + (1.0 - STABILITY_MONTHLY_W) * Weekly_Score

    # ---- Step 6: component scores -------------------------------------------
    Expectancy_Score = _clip(50.0 + 50.0 * Expectancy_R / EXPECTANCY_ANCHOR_R, 0.0, 100.0)
    RR_Score         = 0.0 if math.isnan(RR) else _clip(100.0 * RR / RR_ANCHOR, 0.0, 100.0)
    Streak_Score     = 100.0 * math.exp(-Excess_Streak / STREAK_DECAY)

    Avg_Monthly_PnL = _mean(monthly_buckets)
    if Avg_Monthly_PnL > -RETURN_ANCHOR_INR:
        Return_Score = max(
            0.0,
            100.0 * math.log(1.0 + Avg_Monthly_PnL / RETURN_ANCHOR_INR) / math.log(2.0),
        )
    else:
        Return_Score = 0.0

    # ---- Step 7: combine (NOT clipped) --------------------------------------
    WE = (
        WEIGHT_EXPECTANCY * Expectancy_Score
        + WEIGHT_RR         * RR_Score
        + WEIGHT_STREAK     * Streak_Score
        + WEIGHT_RETURN     * Return_Score
        + WEIGHT_STABILITY  * Stability_Score
    )

    # ---- Step 9: max realized drawdown (informational, not scored) ----------
    equity = peak = start_capital
    mdd = 0.0
    for p in pnl:
        equity += p
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            if dd > mdd:
                mdd = dd

    # ---- Step 8: assemble output row ----------------------------------------
    period_start = rows[0]["exit_date"]  if rows else None
    period_end   = rows[-1]["exit_date"] if rows else None

    def pct(num, den):
        return (100.0 * num / den) if den else 0.0

    return {
        # 8a. Reference metrics (raw)
        "Dataset":                                        label,
        "Period start":                                   period_start,
        "Period end":                                     period_end,
        "Completed trades":                               N,
        "Winning trades":                                 W,
        "Losing trades":                                  L,
        "Success percentage of signal (%)":               round(pct(W, N), 4),
        "Failure percentage (%)":                         round(pct(L, N), 4),
        "Gross derivative P&L proxy (INR)":               round(Gross_PnL, 2),
        "Charges (INR)":                                  round(Total_Charges, 2),
        "Net P&L after charges (INR)":                    round(Net_PnL, 2),
        "Reference invested capital (INR)":               round(start_capital, 2),
        "Overall increase/decrease in reference capital (%)": round(pct(Net_PnL, start_capital), 4),
        "Ending reference capital (INR)":                 round(start_capital + Net_PnL, 2),
        "Average monthly net P&L (INR)":                  round(Avg_Monthly_PnL, 2),
        "Maximum realized drawdown (%)":                  round(mdd, 4),
        # 8b. Raw inputs to the score
        "Average winning trade (INR)":                    round(Avg_Win, 2),
        "Average losing trade absolute (INR)":            round(Avg_Loss, 2),
        "Payoff ratio (RR)":                              "N/A" if math.isnan(RR) else round(RR, 4),
        "Expectancy (R per trade)":                       round(Expectancy_R, 4),
        "Max losing streak":                              int(Max_Streak),
        "Avg losing streak":                              round(Avg_Streak, 4),
        "Excess losing streak":                           round(Excess_Streak, 4),
        "Monthly stability score (/100)":                 round(Monthly_Score, 4),
        "Weekly stability score (/100)":                  round(Weekly_Score, 4),
        # 8c. Component scores
        "Expectancy score (/100)":                        round(Expectancy_Score, 4),
        "Payoff ratio score (/100)":                      round(RR_Score, 4),
        "Loss-streak score (/100)":                       round(Streak_Score, 4),
        "Return score (/100)":                            round(Return_Score, 4),
        "Stability score (/100)":                         round(Stability_Score, 4),
        # 8d. Final
        "Weighted efficiency score (/100)":               round(WE, 4),
        # diagnostics
        "_edge_case":           rr_edge_case,
        "_num_monthly_buckets": len(monthly_buckets),
        "_num_weekly_buckets":  len(weekly_buckets),
    }


# ---------------------------------------------------------------------------
# Convenience loaders — point at ONE file
# ---------------------------------------------------------------------------
def compute_we_from_csv(
    csv_path: str,
    *,
    label: str | None = None,
    start_capital: float = 1_000_000.0,
) -> dict:
    """Read a CSV (columns: pnl, exit_date, [exit_time, gross_pnl, charges]) and score it."""
    trades = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            trades.append({k: v for k, v in row.items() if v not in (None, "")})
    return compute_we(
        trades,
        label=label or csv_path.rsplit("/", 1)[-1],
        start_capital=start_capital,
    )


def compute_we_from_dataframe(
    df,
    *,
    label: str = "All",
    start_capital: float = 1_000_000.0,
) -> dict:
    """Score a pandas DataFrame with the same columns as the CSV loader."""
    return compute_we(
        df.to_dict(orient="records"),
        label=label,
        start_capital=start_capital,
    )


# ---------------------------------------------------------------------------
# Validation against the worked example (Section 11) + edge cases (Section 12)
# ---------------------------------------------------------------------------
def _run_validation() -> None:
    # --- Component math reproducibility of §11 ------------------------------
    Win_Rate = 258 / 359
    RR       = 4907.79 / 9762.13
    Exp_R    = Win_Rate * RR - (1 - Win_Rate)
    Excess   = max(0.0, 4 - 1.3117)

    exp_score    = _clip(50 + 50 * Exp_R / EXPECTANCY_ANCHOR_R, 0, 100)
    rr_score     = _clip(100 * RR / RR_ANCHOR, 0, 100)
    streak_score = 100 * math.exp(-Excess / STREAK_DECAY)
    ret_score    = 100 * math.log(1 + 46_705.83 / RETURN_ANCHOR_INR) / math.log(2)
    stab_score   = 0.5 * 32.5391 + 0.5 * 16.9588
    we = (0.30 * exp_score + 0.20 * rr_score + 0.20 * streak_score
          + 0.15 * ret_score + 0.15 * stab_score)

    expected = {
        "Expectancy_R":    0.0800,
        "Expectancy_Score": 63.33,
        "RR_Score":         33.52,
        "Streak_Score":     40.82,
        "Return_Score":    152.02,
        "Stability_Score":  24.75,
        "WE":            60.380,
    }
    got = {
        "Expectancy_R":    round(Exp_R, 4),
        "Expectancy_Score": round(exp_score, 2),
        "RR_Score":         round(rr_score, 2),
        "Streak_Score":     round(streak_score, 2),
        "Return_Score":     round(ret_score, 2),
        "Stability_Score":  round(stab_score, 2),
        "WE":            round(we, 3),
    }
    print("Worked example validation (§11):")
    for k, exp in expected.items():
        ok = abs(got[k] - exp) < 0.01
        print(f"  {k:18s} expected={exp:>10}  got={got[k]:>10}  {'OK' if ok else 'MISMATCH'}")
        assert ok

    # --- Edge cases (§12) ---------------------------------------------------
    r = compute_we([])
    assert r["Completed trades"] == 0
    assert r["Weighted efficiency score (/100)"] == 0.0

    r = compute_we([
        {"pnl": 100, "exit_date": "2024-01-02"},
        {"pnl": 200, "exit_date": "2024-01-03"},
    ])
    assert r["Payoff ratio (RR)"] == "N/A"
    assert r["Payoff ratio score (/100)"] == 0.0
    assert r["Excess losing streak"] == 0.0
    assert r["Loss-streak score (/100)"] == 100.0

    r = compute_we([
        {"pnl": -100, "exit_date": "2024-01-02"},
        {"pnl":  -50, "exit_date": "2024-01-03"},
    ])
    assert r["Expectancy (R per trade)"] == -1.0
    assert r["Expectancy score (/100)"] == 0.0

    r = compute_we([
        {"pnl":  500, "exit_date": "2024-01-02"},
        {"pnl": -100, "exit_date": "2024-02-05"},
        {"pnl":  300, "exit_date": "2024-03-07"},
    ])
    for k in ["Expectancy score (/100)", "Payoff ratio score (/100)",
              "Loss-streak score (/100)", "Stability score (/100)",
              "Monthly stability score (/100)", "Weekly stability score (/100)"]:
        assert 0.0 <= r[k] <= 100.0, (k, r[k])
    assert r["Return score (/100)"] >= 0.0
    assert r["Weighted efficiency score (/100)"] >= 0.0

    print("\nAll §12 edge-case checks passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    _run_validation()

    if len(sys.argv) >= 2:
        path = sys.argv[1]
        print(f"\nScoring file: {path}")
        result = compute_we_from_csv(path)
        print("\n--- Output row (§8) ---")
        for k, v in result.items():
            if k.startswith("_"):
                continue
            print(f"  {k:52s} {v}")
        print(f"\nDiagnostics: edge_case={result['_edge_case']}, "
              f"monthly_buckets={result['_num_monthly_buckets']}, "
              f"weekly_buckets={result['_num_weekly_buckets']}")
    else:
        # No file given — show a tiny demo on an in-memory list.
        demo = [
            {"pnl":  1200, "exit_date": "2024-01-05", "exit_time": "15:20:00"},
            {"pnl":  -800, "exit_date": "2024-01-09", "exit_time": "15:20:00"},
            {"pnl":  1500, "exit_date": "2024-01-15", "exit_time": "15:20:00"},
            {"pnl":  -600, "exit_date": "2024-02-02", "exit_time": "15:20:00"},
            {"pnl":  2000, "exit_date": "2024-02-08", "exit_time": "15:20:00"},
            {"pnl":  -900, "exit_date": "2024-02-20", "exit_time": "15:20:00"},
            {"pnl":  1100, "exit_date": "2024-03-01", "exit_time": "15:20:00"},
            {"pnl":  1300, "exit_date": "2024-03-11", "exit_time": "15:20:00"},
        ]
        r = compute_we(demo, label="Demo", start_capital=1_000_000)
        print("\nDemo row (8 synthetic trades):")
        for k, v in r.items():
            if not k.startswith("_"):
                print(f"  {k:52s} {v}")
