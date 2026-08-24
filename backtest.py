"""
CCI 반등 + 200일선 하단 + 피보나치 되돌림 — 백테스트
============================================================
대상 : backtest/test_us_stock_data_1000_20, index_chart, coin 의 CSV 차트

[사용법]
  python backtest.py                          # 전체(미국주식+지수+코인)
  python backtest.py --groups us_stock index  # 그룹 선택
  python backtest.py --limit 200              # 앞 200종목만 (빠른 확인)
  python backtest.py --start 2015-01-01 --end 2025-12-31
  python backtest.py --no-portfolio           # 트레이드 통계만

[산출물]  ./bt_output/
  trades_<시각>.csv    : 개별 트레이드 원장
  equity_<시각>.csv    : 실현손익 기준 자산곡선
  summary_<시각>.txt   : 콘솔과 동일한 요약
"""

import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import strategy as ST


DATA_ROOT  = r'C:\Users\user\Documents\claude_coding\backtest'
OUTPUT_DIR = r'./bt_output'

# ── 포트폴리오 시뮬 파라미터 ──
INIT_EQUITY     = 100_000.0
RISK_PER_TRADE  = 0.01    # 1트레이드당 자산 대비 위험
MAX_POS_WEIGHT  = 0.15    # 1종목 최대 비중
MAX_POSITIONS   = 10      # 동시 보유 최대 종목수
MAX_GROSS       = 1.00    # 총 노출 상한

# 신호가 시장 전체 급락 구간에 몰려서 터지는 전략이라, 동시보유 슬롯이 좁으면
# 정작 제일 좋은 구간을 놓치고 한산할 때의 평범한 신호만 담게 된다.
# 그래서 '슬롯을 넓히고 종목당 비중을 줄이는' 방향으로 시나리오를 잡는다.
# (슬롯, 트레이드당 위험, 종목당 최대비중)
PORTFOLIO_SCENARIOS = [(10, 0.010, 0.15), (20, 0.005, 0.08),
                       (40, 0.003, 0.05), (60, 0.002, 0.04)]


def _mkt(mkt, date, col):
    """진입일 기준 시장 국면 조회. 해당 날짜가 없으면 직전 거래일 값."""
    if mkt is None:
        return np.nan
    try:
        idx = mkt.index.searchsorted(pd.Timestamp(date), side='right') - 1
        return mkt[col].values[idx] if idx >= 0 else np.nan
    except Exception:
        return np.nan


# ============================================================
# 트레이드 시뮬레이션
# ============================================================
def simulate_trade(d, i, plan):
    """
    i번째 봉 종가 진입 후 청산까지 진행. 봉 내부 순서는 보수적으로
    (손절 → 1차목표 → SMA200) 로 가정한다.
    """
    o = d['Open'].values;  h = d['High'].values
    l = d['Low'].values;   c = d['Close'].values
    s = d['SMA200'].values; dates = d['Date'].values
    n = len(d)

    entry = plan['entry']; stop = plan['stop']; t1 = plan['t1']
    slip = ST.SLIPPAGE_PCT
    t2_floor = entry * (1.0 + ST.T2_MIN_GAIN_PCT)

    remain = 1.0
    realized = 0.0
    t1_hit = False
    t1_date = None
    mae = 0.0
    mfe = 0.0
    reason = 'open'
    exit_i = None
    exit_px = np.nan

    def net(px):
        return px * (1.0 - slip) / entry - 1.0

    last = min(n - 1, i + ST.MAX_HOLD_BARS)
    for j in range(i + 1, last + 1):
        mae = min(mae, l[j] / entry - 1.0)
        mfe = max(mfe, h[j] / entry - 1.0)

        cur_stop = entry if (t1_hit and ST.BREAKEVEN_AFTER_T1) else stop

        # 1) 손절 (갭하락이면 시가 체결)
        if l[j] <= cur_stop:
            px = min(cur_stop, o[j])
            realized += remain * net(px)
            reason = 'be_stop' if t1_hit else 'stop'
            exit_i, exit_px, remain = j, px, 0.0
            break

        if ST.EXIT_MODE == 'stop_time':
            continue          # 익절 목표 없음 — 손절 아니면 만기까지 보유

        # 2) 1차 익절 — 진입가 위 첫 피보나치 레벨 (갭상승이면 시가 체결)
        if (not t1_hit) and h[j] >= t1:
            px = max(t1, o[j])
            realized += ST.T1_PORTION * net(px)
            remain -= ST.T1_PORTION
            t1_hit = True
            t1_date = dates[j]

        # 3) 2차 익절 — SMA200 살짝 아래 (평균회귀 목표 달성), 단 진입가+2% 하한
        sj = max(s[j] * ST.SMA_TARGET_MULT, t2_floor) if np.isfinite(s[j]) else np.nan
        if np.isfinite(sj) and h[j] >= sj and remain > 0:
            px = max(sj, o[j])
            realized += remain * net(px)
            reason = 'sma200'
            exit_i, exit_px, remain = j, px, 0.0
            break

    if remain > 0:
        j = last
        realized += remain * net(c[j])
        reason = 'time' if j - i >= ST.MAX_HOLD_BARS else 'data_end'
        exit_i, exit_px, remain = j, c[j], 0.0

    return {
        'exit_idx': exit_i, 'exit_date': dates[exit_i], 'exit_price': exit_px,
        'exit_reason': reason, 'ret': realized, 'bars_held': exit_i - i,
        't1_hit': t1_hit, 't1_date': t1_date, 'mae': mae, 'mfe': mfe,
    }


def backtest_symbol(sym, group, df, mkt=None):
    """단일 종목 백테스트 → 트레이드 리스트."""
    d = ST.compute_indicators(df)
    ok = d['ENTRY_OK'].values
    if not ok.any():
        return []

    c = d['Close'].values
    trades = []
    i = 0
    n = len(d)
    while i < n - 1:          # 마지막 봉 신호는 이후 데이터가 없어 집계에서 제외
        if not ok[i]:
            i += 1
            continue
        plan = ST.plan_at(d, i)
        if plan is None:
            i += 1
            continue
        r = simulate_trade(d, i, plan)
        feat = ST.entry_features(d, i)
        trades.append({
            'symbol': sym, 'group': group,
            'entry_date': pd.Timestamp(d['Date'].values[i]).strftime('%Y-%m-%d'),
            'entry_close': round(float(c[i]), 4),
            'entry_price': round(plan['entry'], 4),
            'stop': round(plan['stop'], 4),
            'target1': round(plan['t1'], 4),
            't1_src': plan['t1_src'],
            'risk_pct': round(plan['risk_pct'] * 100, 2),
            'exit_date': pd.Timestamp(r['exit_date']).strftime('%Y-%m-%d'),
            'exit_price': round(float(r['exit_price']), 4),
            'exit_reason': r['exit_reason'],
            'bars_held': int(r['bars_held']),
            't1_hit': bool(r['t1_hit']),
            'ret_pct': round(r['ret'] * 100, 3),
            'r_multiple': round(r['ret'] / plan['risk_pct'], 3),
            'mae_pct': round(r['mae'] * 100, 2),
            'mfe_pct': round(r['mfe'] * 100, 2),
            'cci': round(float(d['CCI'].values[i]), 1),
            'sma200_gap_pct': round((c[i] / d['SMA200'].values[i] - 1) * 100, 2),
            'fib_pos': round(float(d['FIB_POS'].values[i]), 4),
            'fib_low': round(float(d['FIB_LO'].values[i]), 4),
            'fib_high': round(float(d['FIB_HI'].values[i]), 4),
            # 회복세 검증 후보 지표 (거르지 않고 기록만 — filter_lab.py 에서 평가)
            'dvol20': round(feat['dvol20'], 0),
            'vol_ratio': round(feat['vol_ratio'], 3),
            'rsi': round(feat['rsi'], 1),
            'rsi_div': feat['rsi_div'],
            'ma5_gt_ma10': feat['ma5_gt_ma10'],
            'ma_align3': feat['ma_align3'],
            'close_gt_ma5': feat['close_gt_ma5'],
            'atr_pct': round(feat['atr_pct'], 2),
            'bounce_pct': round(feat['bounce_pct'], 2),
            'sma200_room': round(feat['sma200_room'], 2),
            # ── 2차 후보 지표 ──
            'fib_near_pct': round(feat['fib_near_pct'], 3),
            'fib_reclaim': feat['fib_reclaim'],
            'sma200_slope': round(feat['sma200_slope'], 3),
            'days_below200': int(feat['days_below200']),
            'high52_dist': round(feat['high52_dist'], 2),
            'low52_dist': round(feat['low52_dist'], 2),
            'bb_pctb': round(feat['bb_pctb'], 3),
            'sma50_room': round(feat['sma50_room'], 2),
            'cci_min': round(feat['cci_min'], 0),
            'vol_dryup': round(feat['vol_dryup'], 3),
            'higher_low': feat['higher_low'],
            # 시장 국면 (SPY 기준)
            'mkt_above200': _mkt(mkt, d['Date'].values[i], 'MKT_ABOVE200'),
            'mkt_slope': _mkt(mkt, d['Date'].values[i], 'MKT_SLOPE'),
            'mkt_dd': _mkt(mkt, d['Date'].values[i], 'MKT_DD'),
        })
        # 청산 다음 봉부터 재진입 탐색 (포지션 중복 금지)
        i = r['exit_idx'] + 1

    return trades


# ============================================================
# 포트폴리오 시뮬 (실현손익 기준)
# ============================================================
def simulate_portfolio(tr, max_positions=MAX_POSITIONS,
                       risk_per_trade=RISK_PER_TRADE, max_pos_weight=MAX_POS_WEIGHT):
    """
    트레이드를 날짜순으로 흘리며 동시보유/노출 제한을 적용.
    자산곡선은 청산(실현) 시점 기준이라 일별 평가손익은 반영하지 않는다.

    같은 날 신호가 슬롯보다 많으면 위험폭(risk_pct)이 작은 순으로 채운다.
    (같은 1% 위험으로 더 큰 R을 노릴 수 있는 후보 우선 — 사후 성과가 아니라
     사전에 알 수 있는 정보만 사용)
    """
    if tr.empty:
        return pd.DataFrame(), {}

    t = tr.sort_values(['entry_date', 'risk_pct', 'symbol']).reset_index(drop=True)
    events = []
    for i, row in t.iterrows():
        events.append((row['entry_date'], 1, i))    # 1 = 진입
        events.append((row['exit_date'], 0, i))     # 0 = 청산 (같은 날이면 청산 먼저)
    events.sort(key=lambda e: (e[0], e[1]))

    equity = INIT_EQUITY
    deployed = 0.0
    open_pos = {}
    skipped = 0
    curve = [(t['entry_date'].min(), equity)]

    for date, kind, i in events:
        if kind == 0:
            if i in open_pos:
                alloc = open_pos.pop(i)
                deployed -= alloc
                equity += alloc * (t.at[i, 'ret_pct'] / 100.0)
                curve.append((date, equity))
        else:
            rp = max(t.at[i, 'risk_pct'] / 100.0, 1e-6)
            w = min(risk_per_trade / rp, max_pos_weight)
            alloc = equity * w
            if len(open_pos) >= max_positions or deployed + alloc > equity * MAX_GROSS:
                skipped += 1
                continue
            open_pos[i] = alloc
            deployed += alloc

    eq = pd.DataFrame(curve, columns=['date', 'equity'])
    eq = eq.groupby('date', as_index=False).last()
    eq['peak'] = eq['equity'].cummax()
    eq['dd'] = eq['equity'] / eq['peak'] - 1.0

    d0 = pd.Timestamp(eq['date'].iloc[0]); d1 = pd.Timestamp(eq['date'].iloc[-1])
    years = max((d1 - d0).days / 365.25, 1e-6)
    total = equity / INIT_EQUITY - 1.0
    stat = {
        'start': eq['date'].iloc[0], 'end': eq['date'].iloc[-1],
        'final_equity': equity, 'total_return_pct': total * 100,
        'cagr_pct': ((equity / INIT_EQUITY) ** (1 / years) - 1) * 100,
        'mdd_pct': eq['dd'].min() * 100,
        'taken': len(t) - skipped, 'skipped': skipped,
        'fill_pct': (len(t) - skipped) / len(t) * 100,
        'risk': risk_per_trade, 'maxw': max_pos_weight,
    }
    return eq, stat


# ============================================================
# 통계 / 리포트
# ============================================================
def trade_stats(t):
    if t.empty:
        return {}
    r = t['ret_pct']
    wins = r[r > 0]; loss = r[r <= 0]
    gp = wins.sum(); gl = -loss.sum()
    return {
        'trades': len(t),
        'symbols': t['symbol'].nunique(),
        'win_rate': len(wins) / len(t) * 100,
        'avg_ret': r.mean(),
        'median_ret': r.median(),
        'avg_win': wins.mean() if len(wins) else 0.0,
        'avg_loss': loss.mean() if len(loss) else 0.0,
        'expectancy_r': t['r_multiple'].mean(),
        'profit_factor': (gp / gl) if gl > 0 else float('inf'),
        'avg_bars': t['bars_held'].mean(),
        'best': r.max(), 'worst': r.min(),
        't1_hit_rate': t['t1_hit'].mean() * 100,
    }


def _fmt_stats(label, s):
    if not s:
        return f"  {label:<12} (트레이드 없음)\n"
    pf = '∞' if s['profit_factor'] == float('inf') else f"{s['profit_factor']:.2f}"
    return (f"  {label:<12} {s['trades']:>6}건 | 승률 {s['win_rate']:5.1f}% | "
            f"평균 {s['avg_ret']:+6.2f}% | 중앙 {s['median_ret']:+6.2f}% | "
            f"기대값 {s['expectancy_r']:+5.2f}R | PF {pf:>5} | "
            f"평균보유 {s['avg_bars']:4.1f}봉\n")


def build_report(t, eq, stats, elapsed, scanned, matched):
    L = []
    A = L.append
    A("=" * 96)
    A(f"CCI 반등 + 200일선 하단 + 피보나치 되돌림 {ST.FIB_MIN_RETRACE:.2f}~{ST.FIB_MAX_RETRACE:.3f}"
      f" (고점=0/저점=1) — 백테스트 결과")
    A("=" * 96)
    A("")
    A("[진입 조건]")
    A(f"  · 종가 < SMA{ST.SMA_PERIOD}")
    A(f"  · CCI({ST.CCI_PERIOD}) {ST.CCI_OVERSOLD:.0f} 이하 눌림 → {ST.CCI_RECOVER:.0f} 상향 돌파,"
      f" 돌파가 최근 {ST.RECOVER_WINDOW}거래일 내 (눌림~회복 간격 ≤ {ST.DIP_MAX_BARS}봉)")
    anchor = (f"ZigZag {ST.SWING_PCT*100:.0f}% 최근 상승파동" if ST.FIB_MODE == 'swing'
              else f"최대 {ST.FIB_YEARS}년 고저")
    A(f"  · {anchor} 고점(0.0)/저점(1.0) 피보나치에서 종가가"
      f" {ST.FIB_MAX_RETRACE} 레벨 위 (되돌림 깊이 {ST.FIB_MIN_RETRACE}~{ST.FIB_MAX_RETRACE})")
    A("  · 조건 충족 봉의 종가로 진입")
    ex = []
    if ST.USE_ROOM_FILTER:
        ex.append(f"SMA200까지 여유 ≥ {ST.MIN_SMA200_ROOM:.0f}%")
    if ST.USE_ATR_FILTER:
        ex.append(f"ATR ≤ 종가의 {ST.MAX_ATR_PCT:.0f}%")
    if ST.USE_DRYUP_FILTER:
        ex.append(f"거래량 마름 ≤ {ST.MAX_VOL_DRYUP}")
    if ST.USE_FIBNEAR_FILTER:
        ex.append(f"피보 레벨 {ST.MAX_FIB_NEAR_PCT:.0f}% 이내")
    if ST.USE_CLOSE_GT_MA5:
        ex.append("종가 > SMA5")
    if ST.USE_RSI_DIV_FILTER:
        ex.append("RSI 상승 다이버전스")
    A(f"  · [추가 필터] {' + '.join(ex) if ex else '없음'}")
    A("")
    A("[청산 전략]")
    A(f"  · 손절     : 최근 {ST.SWING_LOOKBACK}봉 스윙 저점 -{ST.SWING_BUFFER*100:.1f}%,"
      f" 위험폭 {ST.MIN_STOP_PCT*100:.0f}~{ST.MAX_STOP_PCT*100:.0f}% 로 제한")
    A(f"  · 1차 익절 : 진입가 위 첫 피보나치 레벨 ({ST.T1_PORTION*100:.0f}% 청산,"
      f" {ST.T1_MIN_R:.1f}R~{ST.T1_MAX_R:.1f}R 로 클램프)")
    if ST.EXIT_MODE == 'stop_time':
        A("  · 익절 목표 없음 — 손절에 걸리지 않으면 기간 만료까지 보유")
    else:
        A(f"  · 2차 익절 : SMA{ST.SMA_PERIOD} × {ST.SMA_TARGET_MULT} 도달 시 잔량 청산")
    A(f"  · 1차 익절 후 잔량 손절 = 본전 / 최대 보유 {ST.MAX_HOLD_BARS}봉 / 편도 비용 {ST.SLIPPAGE_PCT*100:.2f}%")
    A("")
    A(f"[스캔] 파일 {scanned}개 중 신호 발생 {matched}개 · 소요 {elapsed:.1f}초")
    A("")

    if t.empty:
        A("트레이드가 발생하지 않았습니다.")
        return "\n".join(L)

    A("-" * 96)
    A("[전체 성과]")
    A(_fmt_stats('ALL', trade_stats(t)).rstrip())
    s = trade_stats(t)
    A(f"    최고 {s['best']:+.2f}% / 최악 {s['worst']:+.2f}% / "
      f"평균이익 {s['avg_win']:+.2f}% / 평균손실 {s['avg_loss']:+.2f}% / "
      f"1차목표 도달률 {s['t1_hit_rate']:.1f}% / 종목수 {s['symbols']}")
    A("")

    A("[그룹별]")
    for g, sub in t.groupby('group'):
        A(_fmt_stats(g, trade_stats(sub)).rstrip())
    A("")

    A("[청산 사유별]")
    for reason, sub in t.groupby('exit_reason'):
        A(f"  {reason:<12} {len(sub):>6}건 ({len(sub)/len(t)*100:5.1f}%) | "
          f"평균 {sub['ret_pct'].mean():+6.2f}% | 평균보유 {sub['bars_held'].mean():4.1f}봉")
    A("")

    A("[연도별 (진입일 기준)]")
    yr = t.copy()
    yr['year'] = yr['entry_date'].str[:4]
    for y, sub in yr.groupby('year'):
        A(f"  {y}         {len(sub):>6}건 | 승률 {(sub['ret_pct']>0).mean()*100:5.1f}% | "
          f"평균 {sub['ret_pct'].mean():+6.2f}% | 합계R {sub['r_multiple'].sum():+7.1f}")
    A("")

    A("[진입 시점 피보나치 위치별]")
    lo, hi = ST.FIB_MIN_RETRACE, ST.FIB_MAX_RETRACE
    bins = [lo] + [e for e in (0.28, 0.36, 0.45, 0.55) if lo < e < hi] + [hi + 1e-9]
    labels = [f'{bins[k]:.3f}~{bins[k+1]:.2f}' for k in range(len(bins) - 1)]
    fp = pd.cut(t['fib_pos'], bins=bins, labels=labels, right=False)
    for lb, sub in t.groupby(fp, observed=True):
        if len(sub) == 0:
            continue
        A(f"  {str(lb):<12} {len(sub):>6}건 | 승률 {(sub['ret_pct']>0).mean()*100:5.1f}% | "
          f"평균 {sub['ret_pct'].mean():+6.2f}% | 기대값 {sub['r_multiple'].mean():+5.2f}R")
    A("")

    if stats:
        A("-" * 96)
        A(f"[포트폴리오 시뮬] 초기 {INIT_EQUITY:,.0f} · 총노출 상한 {MAX_GROSS*100:.0f}%")
        A("  이 전략은 신호가 시장 전체 급락 구간에 몰려서 터진다. 그때가 가장 좋은 구간인데")
        A("  슬롯이 좁으면 그 구간을 몇 개밖에 못 담고, 한산할 때의 평범한 신호만 채우게 된다.")
        A("  → 슬롯을 넓히고 종목당 비중을 줄이는 쪽이 유리하다.")
        A(f"  {'슬롯':<6}{'위험/건':>8}{'최대비중':>9}{'체결':>8}{'체결률':>8}"
          f"{'총수익':>11}{'CAGR':>9}{'MDD':>9}")
        for mp, s in stats:
            A(f"  {mp:<6}{s['risk']*100:>7.2f}%{s['maxw']*100:>8.0f}%{s['taken']:>8}"
              f"{s['fill_pct']:>7.0f}%{s['total_return_pct']:>+10.1f}%"
              f"{s['cagr_pct']:>+8.2f}%{s['mdd_pct']:>8.1f}%")
        A("  (자산곡선은 실현손익 기준 — 보유 중 평가손익은 반영하지 않음)")
        A("")

    if not eq.empty:
        A(f"[연도별 자산 (동시 {stats[-1][0]}종목 기준)]")
        e = eq.copy()
        e['year'] = e['date'].astype(str).str[:4]
        last = e.groupby('year')['equity'].last()
        prev = INIT_EQUITY
        for y, v in last.items():
            A(f"  {y}          {v:>12,.0f}   ({v/prev-1:+7.1%})")
            prev = v
        A("")

    A("[R 합계 상위 종목 15]")
    top = t.groupby('symbol').agg(n=('r_multiple', 'size'), sumR=('r_multiple', 'sum'),
                                  avg=('ret_pct', 'mean')).sort_values('sumR', ascending=False)
    for sym, row in top.head(15).iterrows():
        A(f"  {sym:<10} {int(row['n']):>3}건 | 합계 {row['sumR']:+7.1f}R | 평균 {row['avg']:+6.2f}%")
    A("")
    A("[R 합계 하위 종목 10]")
    for sym, row in top.tail(10).iloc[::-1].iterrows():
        A(f"  {sym:<10} {int(row['n']):>3}건 | 합계 {row['sumR']:+7.1f}R | 평균 {row['avg']:+6.2f}%")
    A("=" * 96)
    return "\n".join(L)


# ============================================================
# 메인
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--groups', nargs='+', default=['us_stock', 'index', 'coin'],
                    help='us_stock index coin kr_stock us_week')
    ap.add_argument('--limit', type=int, default=0, help='그룹별 앞 N개 파일만')
    ap.add_argument('--start', default='', help='YYYY-MM-DD 이후 진입만 집계')
    ap.add_argument('--end', default='', help='YYYY-MM-DD 이전 진입만 집계')
    ap.add_argument('--coin-all-tf', action='store_true', help='코인 일봉 외 타임프레임도 포함')
    ap.add_argument('--no-clean', action='store_true', help='오류틱 정제 끄기(원본 그대로)')
    # 파라미터 오버라이드 — 항목별 효과를 분리해서 보기 위한 스윕용
    ap.add_argument('--fib-min', type=float, default=None, help='최소 되돌림 깊이')
    ap.add_argument('--fib-max', type=float, default=None, help='최대 되돌림 깊이')
    ap.add_argument('--hold', type=int, default=None, help='최대 보유 봉수')
    ap.add_argument('--t1-max-r', type=float, default=None)
    ap.add_argument('--t2-mult', type=float, default=None, help='2차 익절 = SMA200 × 이 값')
    ap.add_argument('--t2-floor', type=float, default=None, help='2차 목표 최소 수익률(0.02=2%%)')
    ap.add_argument('--room', type=float, default=None, help='SMA200까지 최소 여유 %%')
    ap.add_argument('--atr-max', type=float, default=None, help='최대 ATR%%')
    ap.add_argument('--no-filters', action='store_true', help='추가 진입 필터 전부 끄기')
    ap.add_argument('--rsi-div', action='store_true', help='RSI 상승 다이버전스 필터 켜기')
    ap.add_argument('--dryup', type=float, default=None, help='거래량 마름 상한(5일/20일)')
    ap.add_argument('--fib-near', type=float, default=None, help='피보 레벨까지 최대 거리 %%')
    ap.add_argument('--relaxed', action='store_true',
                    help='신호를 늘린 완화 프리셋 (dryup 1.1 / fib 3%%)')
    ap.add_argument('--fib-mode', choices=['range', 'swing'], default=None,
                    help='range=10년 고저, swing=최근 상승 파동')
    ap.add_argument('--swing-pct', type=float, default=None, help='파동 인정 변동폭')
    ap.add_argument('--exit-mode', choices=['full', 'stop_time'], default=None,
                    help='full=목표 익절 포함, stop_time=손절+기간만')
    ap.add_argument('--tag', default='', help='산출 파일명에 붙일 라벨')
    ap.add_argument('--no-portfolio', action='store_true')
    ap.add_argument('--slots', nargs='+', type=int, default=None,
                    help='동시보유 종목수 직접 지정 (기본: 슬롯·비중 조합 시나리오)')
    ap.add_argument('--out', default=OUTPUT_DIR)
    args = ap.parse_args()

    for attr, val in [('FIB_MIN_RETRACE', args.fib_min), ('FIB_MAX_RETRACE', args.fib_max),
                      ('MAX_HOLD_BARS', args.hold), ('T1_MAX_R', args.t1_max_r),
                      ('SMA_TARGET_MULT', args.t2_mult), ('T2_MIN_GAIN_PCT', args.t2_floor),
                      ('MIN_SMA200_ROOM', args.room), ('MAX_ATR_PCT', args.atr_max),
                      ('MAX_VOL_DRYUP', args.dryup), ('MAX_FIB_NEAR_PCT', args.fib_near),
                      ('EXIT_MODE', args.exit_mode), ('FIB_MODE', args.fib_mode),
                      ('SWING_PCT', args.swing_pct)]:
        if val is not None:
            setattr(ST, attr, val)
            print(f"  [override] {attr} = {val}")
    if args.relaxed:
        ST.MAX_VOL_DRYUP, ST.MAX_FIB_NEAR_PCT = 1.1, 3.0
        print("  [override] 완화 프리셋: dryup 1.1 / fib_near 3%")
    if args.no_filters:
        ST.USE_ROOM_FILTER = ST.USE_ATR_FILTER = ST.USE_RSI_DIV_FILTER = False
        ST.USE_DRYUP_FILTER = ST.USE_FIBNEAR_FILTER = ST.USE_CLOSE_GT_MA5 = False
        print("  [override] 추가 진입 필터 전부 off")
    if args.rsi_div:
        ST.USE_RSI_DIV_FILTER = True
        print("  [override] RSI 다이버전스 필터 on")

    files = ST.collect_csv_files(args.data_root, args.groups,
                                 coin_daily_only=not args.coin_all_tf)
    if args.limit:
        keep, cnt = [], {}
        for sym, g, p in files:
            cnt[g] = cnt.get(g, 0) + 1
            if cnt[g] <= args.limit:
                keep.append((sym, g, p))
        files = keep

    if not files:
        print("[ERROR] 대상 CSV가 없습니다."); sys.exit(1)

    mkt = ST.load_market_context(args.data_root)
    print(f"  시장 국면(SPY) {'로드 완료' if mkt is not None else '없음'}")
    print(f"대상 파일 {len(files)}개 — 백테스트 시작")
    t0 = time.time()
    all_trades = []
    matched = 0
    for k, (sym, g, path) in enumerate(files, 1):
        df = ST.load_csv(path, clean=not args.no_clean)
        if df is not None and len(df) >= ST.MIN_BARS:
            try:
                tr = backtest_symbol(sym, g, df, mkt)
                if tr:
                    matched += 1
                    all_trades.extend(tr)
            except Exception as e:
                print(f"  [WARN] {sym} 실패: {e}")
        if k % 200 == 0 or k == len(files):
            print(f"  {k}/{len(files)} ... 누적 트레이드 {len(all_trades)}건 "
                  f"({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    t = pd.DataFrame(all_trades)
    if not t.empty:
        if args.start:
            t = t[t['entry_date'] >= args.start]
        if args.end:
            t = t[t['entry_date'] <= args.end]
        t = t.sort_values(['entry_date', 'symbol']).reset_index(drop=True)

    eq, stats = (pd.DataFrame(), [])
    if not args.no_portfolio and not t.empty:
        scen = ([(mp, RISK_PER_TRADE, MAX_POS_WEIGHT) for mp in args.slots]
                if args.slots else PORTFOLIO_SCENARIOS)
        for mp, risk, maxw in scen:
            e, s = simulate_portfolio(t, max_positions=mp,
                                      risk_per_trade=risk, max_pos_weight=maxw)
            if s:
                stats.append((mp, s))
                eq = e           # 마지막(가장 넓은) 시나리오의 곡선을 저장

    report = build_report(t, eq, stats, elapsed, len(files), matched)
    print()
    print(report)

    os.makedirs(args.out, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S') + (f'_{args.tag}' if args.tag else '')
    if not t.empty:
        tp = os.path.join(args.out, f'trades_{ts}.csv')
        t.to_csv(tp, index=False, encoding='utf-8-sig')
        print(f"\n저장: {tp}")
    if not eq.empty:
        ep = os.path.join(args.out, f'equity_{ts}.csv')
        eq.to_csv(ep, index=False, encoding='utf-8-sig')
        print(f"저장: {ep}")
    sp = os.path.join(args.out, f'summary_{ts}.txt')
    with open(sp, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"저장: {sp}")


if __name__ == '__main__':
    main()
