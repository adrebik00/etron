"""
무작위 진입 대조군 (Random Entry Benchmark)
============================================================
"손절 걸고 N일 보유" 청산이 정말 우위가 있는지 확인하려면, 같은 청산 규칙을
**아무 종목에나 아무 날에** 적용했을 때와 비교해야 한다. 그게 이 스크립트다.

핵심은 **날짜 매칭**이다. 실제 신호가 난 날짜 분포를 그대로 써서 무작위 진입을
뽑는다. 이걸 안 하면 시장 상황이 다른 기간끼리 비교하게 되어 무의미하다.

대조군 3종:
  R1 무작위 + 같은 손절 + N일 보유   ← 전략의 '진입'만 무작위로 바꾼 것
  R2 무작위 + 손절 없이 N일 보유     ← 순수 매수후보유
  R3 전략 진입 + 손절 없이 N일 보유  ← 손절이 기여하는 몫

[사용법]
  python random_bench.py --trades bt_output/trades_xxx.csv --hold 30 --draws 20
"""
import os, glob, argparse
import numpy as np
import pandas as pd
import strategy as ST

DATA_ROOT = r'C:\Users\user\Documents\claude_coding\backtest'


def load_universe(data_root, group='us_stock'):
    """{sym: dict(dates=DatetimeIndex, o,h,l,c=ndarray)} 로 통째로 메모리에."""
    files = ST.collect_csv_files(data_root, [group])
    uni = {}
    for k, (sym, g, path) in enumerate(files, 1):
        df = ST.load_csv(path)
        if df is None or len(df) < ST.MIN_BARS:
            continue
        uni[sym] = dict(dates=pd.DatetimeIndex(df['Date'].values),
                        o=df['Open'].values.astype(float),
                        h=df['High'].values.astype(float),
                        l=df['Low'].values.astype(float),
                        c=df['Close'].values.astype(float))
        if k % 400 == 0:
            print(f"  {k}/{len(files)} 로드...", flush=True)
    print(f"  유니버스 {len(uni)}종목")
    return uni


def make_plan(u, i):
    """전략과 동일한 손절 규칙 (최근 10봉 스윙 저점, 위험폭 3~12% 클램프)."""
    lo = u['l'][max(0, i - ST.SWING_LOOKBACK + 1): i + 1].min()
    entry = u['c'][i] * (1 + ST.SLIPPAGE_PCT)
    stop = lo * (1 - ST.SWING_BUFFER)
    stop = max(stop, entry * (1 - ST.MAX_STOP_PCT))
    stop = min(stop, entry * (1 - ST.MIN_STOP_PCT))
    risk = entry - stop
    return (entry, stop, risk / entry) if risk > 0 else None


def run_trade(u, i, hold, use_stop=True):
    """손절(옵션) + hold봉 만기 청산. 반환 (수익률, R배수, 청산사유)."""
    p = make_plan(u, i)
    if p is None:
        return None
    entry, stop, risk_frac = p
    n = len(u['c'])
    last = min(n - 1, i + hold)
    if last <= i:
        return None
    for j in range(i + 1, last + 1):
        if use_stop and u['l'][j] <= stop:
            px = min(stop, u['o'][j])
            r = px * (1 - ST.SLIPPAGE_PCT) / entry - 1
            return r, r / risk_frac, 'stop'
    r = u['c'][last] * (1 - ST.SLIPPAGE_PCT) / entry - 1
    return r, r / risk_frac, 'time'


def stats(rets, rs):
    rets = np.asarray(rets); rs = np.asarray(rs)
    gp = rets[rets > 0].sum(); gl = -rets[rets <= 0].sum()
    return dict(n=len(rets), win=(rets > 0).mean() * 100, avg=rets.mean() * 100,
                med=np.median(rets) * 100, expR=rs.mean(),
                pf=(gp / gl) if gl > 0 else np.inf, sd=rets.std() * 100)


def fmt(lab, s):
    pf = ' inf' if not np.isfinite(s['pf']) else f"{s['pf']:.2f}"
    return (f"  {lab:<34}{s['n']:>7}{s['win']:>7.1f}%{s['avg']:>+8.2f}%"
            f"{s['med']:>+8.2f}%{s['expR']:>+8.3f}R{pf:>7}{s['sd']:>8.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades', default='')
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--hold', type=int, default=30)
    ap.add_argument('--draws', type=int, default=20, help='실제 신호 1건당 무작위 표본 수')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    path = args.trades or sorted(glob.glob('./bt_output/trades_*.csv'),
                                 key=os.path.getmtime)[-1]
    t = pd.read_csv(path)
    t = t[t.group == 'us_stock']
    print(f"원장: {path}  ({len(t)}건, 보유 {args.hold}봉 기준으로 재시뮬)")

    rng = np.random.default_rng(args.seed)
    print("유니버스 로드 중...")
    uni = load_universe(args.data_root)
    syms = list(uni)

    # 날짜별 전체 목록을 만들면 800만 항목이라 메모리를 못 견딘다.
    # 대신 종목을 무작위로 뽑고 그 종목이 해당 날짜를 갖는지 searchsorted 로 확인한다.
    def draw(dt, k):
        out = []
        tries = 0
        while len(out) < k and tries < k * 6:
            tries += 1
            s2 = syms[rng.integers(0, len(syms))]
            d = uni[s2]['dates']
            j = d.searchsorted(dt)
            if j < len(d) and d[j] == dt and j >= ST.MIN_BARS and j < len(d) - 1:
                out.append((s2, j))
        return out

    sig_r, sig_R = [], []          # 전략 진입 + 손절 + 기간
    signs_r, signs_R = [], []      # 전략 진입 + 손절 없음
    rnd_r, rnd_R = [], []          # 무작위 + 손절 + 기간
    rndns_r, rndns_R = [], []      # 무작위 + 손절 없음
    stop_hit_sig = stop_hit_rnd = 0
    paired = []          # (진입일, 전략수익, 그날 무작위 평균수익)

    for _, row in t.iterrows():
        dt = pd.Timestamp(row.entry_date)
        # 1) 전략 진입 재시뮬
        u = uni.get(row.symbol)
        my = None
        if u is not None:
            k = u['dates'].searchsorted(dt)
            if k < len(u['dates']) and u['dates'][k] == dt:
                a = run_trade(u, k, args.hold, True)
                b = run_trade(u, k, args.hold, False)
                if a:
                    sig_r.append(a[0]); sig_R.append(a[1]); stop_hit_sig += (a[2] == 'stop')
                    my = a[0]
                if b: signs_r.append(b[0]); signs_R.append(b[1])
        # 2) 같은 날짜에 무작위 종목
        day = []
        for s2, k2 in draw(dt, args.draws):
            a = run_trade(uni[s2], k2, args.hold, True)
            b = run_trade(uni[s2], k2, args.hold, False)
            if a:
                rnd_r.append(a[0]); rnd_R.append(a[1]); stop_hit_rnd += (a[2] == 'stop')
                day.append(a[0])
            if b: rndns_r.append(b[0]); rndns_R.append(b[1])
        if my is not None and day:
            paired.append((row.entry_date, my, float(np.mean(day))))

    S  = stats(sig_r, sig_R);      SN = stats(signs_r, signs_R)
    R  = stats(rnd_r, rnd_R);      RN = stats(rndns_r, rndns_R)

    L = []
    A = L.append
    A("=" * 104)
    A(f"청산 = 손절 + {args.hold}거래일 보유  |  무작위 진입 대조군 비교")
    A("=" * 104)
    A(f"진입일 분포를 실제 신호와 동일하게 맞춰 무작위 표본을 뽑았다"
      f" (신호 1건당 {args.draws}개).")
    A("")
    A(f"  {'':<34}{'건수':>7}{'승률':>8}{'평균':>8}{'중앙':>8}{'기대값':>9}{'PF':>7}{'표준편차':>8}")
    A("  " + "-" * 100)
    A(fmt(f"전략 진입 + 손절 + {args.hold}일", S))
    A(fmt(f"무작위 진입 + 손절 + {args.hold}일", R))
    A(fmt(f"전략 진입 + 손절없이 {args.hold}일", SN))
    A(fmt(f"무작위 진입 + 손절없이 {args.hold}일", RN))
    A("")
    A(f"  손절 도달률: 전략 {stop_hit_sig/max(S['n'],1)*100:.1f}% / "
      f"무작위 {stop_hit_rnd/max(R['n'],1)*100:.1f}%")
    A("")
    A("[해석]")
    A(f"  · 진입의 순수 기여(같은 청산 규칙): 평균 {S['avg']-R['avg']:+.2f}%p, "
      f"기대값 {S['expR']-R['expR']:+.3f}R")
    A(f"  · 손절의 기여(전략 진입 기준)    : 평균 {S['avg']-SN['avg']:+.2f}%p, "
      f"기대값 {S['expR']-SN['expR']:+.3f}R")

    # ── 날짜 짝지음 검정 ──
    # 같은 날 시장이 통째로 오르내린 몫을 빼고, '그날 아무거나 산 것보다 나았나'만 본다.
    pdf = pd.DataFrame(paired, columns=['date', 'sig', 'rnd'])
    diff = (pdf.sig - pdf.rnd).values
    nn = len(diff)
    se = diff.std(ddof=1) / np.sqrt(nn)
    tstat = diff.mean() / se if se > 0 else 0.0
    A("[날짜 짝지음 검정]  같은 날 무작위 평균 대비 초과수익")
    A(f"  표본 {nn}쌍 · 평균 초과 {diff.mean()*100:+.2f}%p · "
      f"중앙 {np.median(diff)*100:+.2f}%p · 이긴 비율 {(diff>0).mean()*100:.1f}%")
    A(f"  t = {tstat:.2f}  ->  {'유의미' if abs(tstat) > 2 else '유의하지 않음'} (|t|>2 기준)")
    pdf['yr'] = pdf.date.str[:4]
    yrs = pdf.yr.unique()
    byyr = {y: (pdf[pdf.yr == y].sig - pdf[pdf.yr == y].rnd).values for y in yrs}
    bs = []
    for _ in range(2000):
        pick = rng.choice(yrs, size=len(yrs), replace=True)
        bs.append(np.concatenate([byyr[y] for y in pick]).mean())
    bs = np.asarray(bs)
    A(f"  연도 블록 부트스트랩 95% 구간 [{np.percentile(bs,2.5)*100:+.2f}%p, "
      f"{np.percentile(bs,97.5)*100:+.2f}%p] · 0 초과 확률 {(bs>0).mean()*100:.1f}%")
    A("")
    A("  같은 해 신호들은 서로 독립이 아니라, 연도 단위로 다시 뽑는 부트스트랩이")
    A("  단순 t검정보다 현실적이다. 구간이 0을 포함하면 우위를 입증하지 못한 것이다.")
    A("=" * 104)
    out = "\n".join(L)
    print(); print(out)
    op = f'./bt_output/random_bench_hold{args.hold}.txt'
    open(op, 'w', encoding='utf-8').write(out)
    print(f"\n저장: {op}")


if __name__ == '__main__':
    main()
