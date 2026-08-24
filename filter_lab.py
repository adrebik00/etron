"""
회복세 검증 조건 실험실
============================================================
backtest.py 가 남긴 트레이드 원장(trades_*.csv)을 읽어, 진입 시점에 알 수 있는
후보 조건들이 성과를 실제로 개선하는지 측정한다.

핵심은 "좋아 보이는 조건 찾기"가 아니라 "우연이 아닌지 확인하기"다. 그래서
모든 조건을 **전반기/후반기로 나눠** 양쪽에서 같은 방향으로 작동하는지 본다.
한쪽에서만 좋은 조건은 과최적화로 보고 채택하지 않는다.

[사용법]
  python filter_lab.py                          # 가장 최근 trades_*.csv 사용
  python filter_lab.py --trades bt_output/trades_xxx.csv
  python filter_lab.py --split 2017-01-01       # 전/후반 경계
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd


# ============================================================
# 후보 조건 정의 — 이름: (설명, 판정함수)
# ============================================================
def build_filters():
    """
    전부 '진입 시점에 알 수 있는 값'만 쓴다. 미래 정보 없음.
    시가총액은 과거 시점 값을 구할 수 없어(발행주식수 이력 필요) 20일 평균
    거래대금으로 대신한다 — 유동성/규모 프록시.

    같은 지표의 강도만 다른 변형은 family 를 묶어둔다. 조합 테스트에서
    room_ge8 과 room_ge5 를 AND 하면 아무 의미가 없기 때문(전자가 후자를 포함).
    """
    return {
        # 이름: (설명, 판정함수, family)
        # ── 유동성 / 규모 (시총 대용) ──
        'dvol_1M':   ('20일 평균 거래대금 ≥ $1M',   lambda d: d.dvol20 >= 1e6,     'dvol'),
        'dvol_10M':  ('20일 평균 거래대금 ≥ $10M',  lambda d: d.dvol20 >= 1e7,     'dvol'),
        'dvol_50M':  ('20일 평균 거래대금 ≥ $50M',  lambda d: d.dvol20 >= 5e7,     'dvol'),

        # ── 회복봉에 거래량이 실렸는가 ──
        'vol_1.0':   ('회복봉 거래량 ≥ 20일평균',      lambda d: d.vol_ratio >= 1.0, 'vol'),
        'vol_1.5':   ('회복봉 거래량 ≥ 20일평균×1.5',  lambda d: d.vol_ratio >= 1.5, 'vol'),
        'vol_2.0':   ('회복봉 거래량 ≥ 20일평균×2.0',  lambda d: d.vol_ratio >= 2.0, 'vol'),

        # ── RSI ──
        'rsi_div':   ('RSI 상승 다이버전스(저점상승)', lambda d: d.rsi_div,        'rsidiv'),
        'rsi_ge35':  ('RSI(14) ≥ 35',                 lambda d: d.rsi >= 35,       'rsilvl'),
        'rsi_ge40':  ('RSI(14) ≥ 40',                 lambda d: d.rsi >= 40,       'rsilvl'),

        # ── 단기 이평 회복 ──
        'ma5>ma10':  ('SMA5 > SMA10',                 lambda d: d.ma5_gt_ma10,     'ma'),
        'ma_align3': ('SMA5 > SMA10 > SMA20 정배열',   lambda d: d.ma_align3,       'ma'),
        'c>ma5':     ('종가 > SMA5',                  lambda d: d.close_gt_ma5,    'ma'),

        # ── 변동성 / 반등폭 / 목표까지 여유 ──
        'atr_le6':   ('ATR ≤ 종가의 6%',              lambda d: d.atr_pct <= 6,    'atr'),
        'atr_le4':   ('ATR ≤ 종가의 4%',              lambda d: d.atr_pct <= 4,    'atr'),
        'bounce_le5':('저점 대비 반등 ≤ 5% (덜 뛴 것)', lambda d: d.bounce_pct <= 5, 'bounce'),
        'room_ge3':  ('SMA200까지 여유 ≥ 3%',          lambda d: d.sma200_room >= 3, 'room'),
        'room_ge5':  ('SMA200까지 여유 ≥ 5%',          lambda d: d.sma200_room >= 5, 'room'),
        'room_ge8':  ('SMA200까지 여유 ≥ 8%',          lambda d: d.sma200_room >= 8, 'room'),

        # ── 시장 국면 (SPY) ──
        'mkt_bull':  ('SPY > 자기 SMA200',             lambda d: d.mkt_above200,     'mkt'),
        'mkt_dd_le10':('SPY 1년고점 대비 -10% 이내',    lambda d: d.mkt_dd >= -10,    'mktdd'),
        'mkt_dd_le5':('SPY 1년고점 대비 -5% 이내',      lambda d: d.mkt_dd >= -5,     'mktdd'),
        'mkt_slope+':('SPY SMA200 상승 중',            lambda d: d.mkt_slope > 0,    'mktslp'),

        # ── 피보나치 지지 확인 ──
        'fib_near2': ('피보 레벨에서 2% 이내',          lambda d: d.fib_near_pct <= 2, 'fibnear'),
        'fib_near5': ('피보 레벨에서 5% 이내',          lambda d: d.fib_near_pct <= 5, 'fibnear'),
        'fib_reclaim':('저점이 피보 레벨 이탈 후 종가 회복', lambda d: d.fib_reclaim,  'fibrec'),

        # ── 추세 구조 / 위치 ──
        'slope_ge0': ('SMA200 상승 중',                lambda d: d.sma200_slope >= 0, 'slope'),
        'slope_ge1': ('SMA200 20봉 +1% 이상',          lambda d: d.sma200_slope >= 1, 'slope'),
        'below_le30':('200선 아래 머문 기간 ≤ 30봉',    lambda d: d.days_below200 <= 30, 'below'),
        'below_le60':('200선 아래 머문 기간 ≤ 60봉',    lambda d: d.days_below200 <= 60, 'below'),
        'h52_ge-25': ('52주 고점 대비 -25% 이내',       lambda d: d.high52_dist >= -25, 'h52'),
        'l52_ge10':  ('52주 저점 대비 +10% 이상',       lambda d: d.low52_dist >= 10,  'l52'),
        'higher_low':('직전 저점보다 높은 저점',         lambda d: d.higher_low,        'hl'),
        'sma50_ge3': ('SMA50까지 여유 ≥ 3%',            lambda d: d.sma50_room >= 3,   'sma50'),

        # ── 눌림 깊이 / 거래량 마름 ──
        'cci_le250': ('CCI 최저 ≤ -250 (깊게 눌림)',    lambda d: d.cci_min <= -250,   'ccidepth'),
        'cci_le300': ('CCI 최저 ≤ -300',               lambda d: d.cci_min <= -300,   'ccidepth'),
        'dryup_le09':('최근 5일 거래량 ≤ 20일평균×0.9', lambda d: d.vol_dryup <= 0.9,  'dryup'),
        'bb_le02':   ('볼린저 %B ≤ 0.2',               lambda d: d.bb_pctb <= 0.2,    'bb'),
    }


# ============================================================
# 통계
# ============================================================
def stats(t):
    if len(t) == 0:
        return None
    r = t.ret_pct
    gp = r[r > 0].sum(); gl = -r[r <= 0].sum()
    return {
        'n': len(t),
        'win': (r > 0).mean() * 100,
        'avg': r.mean(),
        'expR': t.r_multiple.mean(),
        'pf': (gp / gl) if gl > 0 else np.inf,
        'sumR': t.r_multiple.sum(),
    }


def _fmt(s):
    if s is None:
        return f"{'-':>8}{'-':>8}{'-':>9}{'-':>8}{'-':>7}"
    pf = ' inf' if not np.isfinite(s['pf']) else f"{s['pf']:.2f}"
    return (f"{s['n']:>8}{s['win']:>7.1f}%{s['avg']:>+8.2f}%"
            f"{s['expR']:>+8.3f}R{pf:>7}")


def evaluate(t, filters, split_date):
    """조건별로 통과분 / 탈락분 성과를 비교하고, 전·후반 안정성까지 본다."""
    base = stats(t)
    early = t[t.entry_date < split_date]
    late = t[t.entry_date >= split_date]
    b_e, b_l = stats(early), stats(late)

    rows = []
    for key, (desc, fn, family) in filters.items():
        try:
            m = fn(t).fillna(False).astype(bool)
        except Exception as e:
            print(f"  [WARN] {key}: {e}")
            continue
        keep, drop = stats(t[m]), stats(t[~m])
        k_e = stats(early[fn(early).fillna(False).astype(bool)])
        k_l = stats(late[fn(late).fillna(False).astype(bool)])
        rows.append({
            'key': key, 'desc': desc, 'family': family, 'keep': keep, 'drop': drop,
            'early': k_e, 'late': k_l,
            'pass_pct': m.mean() * 100,
            'd_exp': (keep['expR'] - base['expR']) if keep else np.nan,
            'd_e': (k_e['expR'] - b_e['expR']) if (k_e and b_e) else np.nan,
            'd_l': (k_l['expR'] - b_l['expR']) if (k_l and b_l) else np.nan,
        })
    return base, b_e, b_l, rows


def report(t, split_date, top_n=8):
    F = build_filters()
    base, b_e, b_l, rows = evaluate(t, F, split_date)
    L = []
    A = L.append

    A("=" * 104)
    A("회복세 검증 조건 실험 — 각 조건을 통과한 트레이드만 남겼을 때의 성과")
    A("=" * 104)
    A(f"원장 {len(t):,}건 · {t.entry_date.min()} ~ {t.entry_date.max()}"
      f" · 전/후반 경계 {split_date}")
    A(f"  전체(필터 없음) {_fmt(base)}")
    A(f"    전반기        {_fmt(b_e)}")
    A(f"    후반기        {_fmt(b_l)}")
    A("")
    A("[조건별 단독 효과]  Δ = 기대값(R) 변화, 전/후반 둘 다 +여야 신뢰")
    A(f"  {'조건':<12}{'통과율':>7}{'건수':>8}{'승률':>8}{'평균':>9}{'기대값':>9}{'PF':>7}"
      f"{'Δ전체':>8}{'Δ전반':>8}{'Δ후반':>8}  설명")
    A("  " + "-" * 100)
    for r in sorted(rows, key=lambda x: -(x['d_exp'] if np.isfinite(x['d_exp']) else -9)):
        both = (np.isfinite(r['d_e']) and np.isfinite(r['d_l'])
                and r['d_e'] > 0 and r['d_l'] > 0)
        mark = '  OK' if both else ''
        A(f"  {r['key']:<12}{r['pass_pct']:>6.0f}%{_fmt(r['keep'])}"
          f"{r['d_exp']:>+8.3f}{r['d_e']:>+8.3f}{r['d_l']:>+8.3f}{mark}  {r['desc']}")
    A("")
    A("  Δ전반·Δ후반이 모두 +인 조건에만 OK 표시. 한쪽만 좋으면 우연일 가능성이 크다.")
    A("")

    # ── 전·후반 모두 개선된 조건만 골라 조합 ──
    good = [r for r in rows
            if np.isfinite(r['d_e']) and np.isfinite(r['d_l'])
            and r['d_e'] > 0 and r['d_l'] > 0 and r['pass_pct'] >= 15]
    good.sort(key=lambda x: -x['d_exp'])
    seen = set()                      # family 당 하나만 — 중첩 조건 AND는 무의미
    good = [r for r in good if not (r['family'] in seen or seen.add(r['family']))]
    A("[조합 테스트] OK 조건(통과율 15%↑)을 기대값 높은 순으로 누적 AND (family당 1개)")
    if not good:
        A("  양쪽에서 모두 개선된 조건이 없다.")
    else:
        A(f"  {'조합':<40}{'건수':>8}{'승률':>8}{'평균':>9}{'기대값':>9}{'PF':>7}{'합계R':>10}")
        A("  " + "-" * 92)
        mask = pd.Series(True, index=t.index)
        names = []
        for r in good[:top_n]:
            mask &= F[r['key']][1](t).fillna(False).astype(bool)   # [1] = 판정함수
            names.append(r['key'])
            s = stats(t[mask])
            if s is None or s['n'] < 50:
                A(f"  {'+'.join(names):<40}  표본 부족({0 if s is None else s['n']}건) — 중단")
                break
            A(f"  {'+'.join(names):<40}{_fmt(s)}{s['sumR']:>+10.0f}")
    A("")
    A("  주의: 여기 나온 개선폭은 같은 데이터에서 고른 것이라 낙관적으로 치우친다.")
    A("  실제 채택은 조건 수를 최소로 줄이고, 표본이 충분한 것만 쓰는 편이 안전하다.")
    A("=" * 104)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades', default='', help='트레이드 원장 CSV (기본: 가장 최근)')
    ap.add_argument('--split', default='2017-01-01', help='전/후반 경계일')
    ap.add_argument('--group', default='us_stock', help='분석 대상 그룹 (all=전체)')
    ap.add_argument('--out', default='./bt_output')
    args = ap.parse_args()

    path = args.trades
    if not path:
        cands = sorted(glob.glob(os.path.join(args.out, 'trades_*.csv')),
                       key=os.path.getmtime)
        if not cands:
            print("[ERROR] trades_*.csv 가 없습니다. backtest.py 를 먼저 실행하세요.")
            return
        path = cands[-1]
    print(f"원장: {path}")

    t = pd.read_csv(path)
    need = ['dvol20', 'vol_ratio', 'rsi', 'rsi_div', 'ma5_gt_ma10',
            'ma_align3', 'close_gt_ma5', 'atr_pct', 'bounce_pct', 'sma200_room',
            'fib_near_pct', 'fib_reclaim', 'sma200_slope', 'days_below200',
            'high52_dist', 'low52_dist', 'bb_pctb', 'sma50_room', 'cci_min',
            'vol_dryup', 'higher_low', 'mkt_above200', 'mkt_slope', 'mkt_dd']
    miss = [c for c in need if c not in t.columns]
    if miss:
        print(f"[ERROR] 피처 컬럼 없음: {miss} — backtest.py 를 다시 실행하세요.")
        return

    if args.group != 'all':
        t = t[t.group == args.group]
    t = t.reset_index(drop=True)
    if t.empty:
        print("[ERROR] 대상 트레이드가 없습니다.")
        return

    rep = report(t, args.split)
    print()
    print(rep)
    op = os.path.join(args.out, 'filter_lab_' +
                      os.path.basename(path).replace('trades_', '').replace('.csv', '.txt'))
    with open(op, 'w', encoding='utf-8') as f:
        f.write(rep)
    print(f"\n저장: {op}")


if __name__ == '__main__':
    main()
