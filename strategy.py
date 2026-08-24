"""
CCI 반등 + 200일선 하단 + 피보나치 되돌림 전략 — 공용 코어
============================================================
backtest.py / monitor.py 가 동일한 신호 정의를 쓰도록 로직을 여기에 모아둔다.

[진입 조건]
  1. 종가 < SMA200                     (일봉 200선 아래)
  2. CCI(20)가 -200 이하로 눌린 뒤 -100 위로 회복,
     그 회복(상향 돌파)이 최근 5거래일 안에 발생
  3. 판단 시점 기준 최대 10년치 차트의 고점(=1.0) / 저점(=0.0)으로 그린
     피보나치 되돌림(고점=0 / 저점=1)에서 종가가 0.618 레벨 위
  → 조건 충족 봉의 종가로 진입

[청산 전략]  (전략 취지 = 200선 아래 과매도 반등 → 평균회귀)
  · 손절   : 직전 스윙 저점 살짝 아래. 단 위험폭은 진입가의 3%~12%로 제한
  · 1차 익절(50%) : 진입가 위 첫 피보나치 레벨. 없거나 너무 가까우면 1.5R,
                    단 3R 초과는 잘라냄 (10년 피보 간격이 과도하게 벌어지는 경우 방어)
  · 2차 익절(잔량) : SMA200 터치 = 평균회귀 목표 달성
  · 1차 익절 후 잔량 손절은 본전(BE)으로 상향
  · 시간 청산 : 최대 60거래일 보유 후 종가 청산
"""

import os
import glob
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ============================================================
# 전략 파라미터
# ============================================================
CCI_PERIOD        = 20
SMA_PERIOD        = 200
ATR_PERIOD        = 14

CCI_OVERSOLD      = -200.0   # 눌림 기준
CCI_RECOVER       = -100.0   # 회복 기준
RECOVER_WINDOW    = 5        # 회복(상향돌파)이 최근 N거래일 안에 발생해야 함
DIP_MAX_BARS      = 5        # -200 터치 ~ -100 회복 사이 허용 간격(봉)

FIB_YEARS         = 10       # 피보나치 기준 최대 기간
# 되돌림은 **고점 = 0.0 / 저점 = 1.0** 기준으로 그린다.
#   레벨가 = 고점 - 비율 × (고점 - 저점)   → 비율이 클수록 낮은 가격
# 따라서 '0.618 위'는 0.618 레벨보다 비싼 구간, 즉 되돌림 깊이가 0.618 미만인 상태.
FIB_MAX_RETRACE   = 0.618    # 이보다 깊게 되돌리면 제외 (가격 하한선)
FIB_MIN_RETRACE   = 0.20     # 이보다 얕으면 제외 — 고점 근처는 SMA200까지 여유가 없어
                             # 승률은 높아도 먹는 폭이 작아 기대값이 낮았다 (가격 상한선)
FIB_RATIOS        = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

# 피보나치 앵커 잡는 방식
#   'range' = 최대 10년 구간의 최고/최저 (단순하지만 종목과 무관한 옛 지점이 잡히기 쉽다)
#   'swing' = ZigZag 로 잡은 **최근 상승 파동**(의미있는 저점 → 고점) 한 개
# 검증 결과 swing 이 range 를 이기지 못해 기본값은 range 로 둔다 (README 5장 참고).
FIB_MODE          = 'range'
SWING_PCT         = 0.20     # 이 정도는 움직여야 파동으로 인정 (반대로 20% 되돌리면 피벗 확정)
SWING_MAX_YEARS   = 5        # 앵커 파동이 이보다 오래되면 무효 처리
SWING_MIN_LEG_PCT = 0.25     # 저점→고점 상승폭이 최소 이만큼은 되어야 의미있는 파동

MIN_BARS          = 260      # 최소 데이터 길이(SMA200 + 여유)
FIB_MIN_OBS       = 250      # 피보나치 산출에 필요한 최소 관측 수

# ── 청산 파라미터 ──
SWING_LOOKBACK    = 10       # 손절용 스윙 저점 탐색 봉수
SWING_BUFFER      = 0.003    # 스윙 저점 아래 여유
MIN_STOP_PCT      = 0.03     # 최소 위험폭
MAX_STOP_PCT      = 0.12     # 최대 위험폭
T1_MIN_GAP_PCT    = 0.03     # 1차 목표 최소 이격
T1_FALLBACK_R     = 1.5      # 피보 레벨이 없을 때 1차 목표
T1_MIN_R          = 1.0      # 1차 목표 최소 R
T1_MAX_R          = 3.0      # 1차 목표 최대 R (필터 강화 후 3.0이 최적)
T1_PORTION        = 0.5      # 1차 익절 비중
BREAKEVEN_AFTER_T1 = True
SMA_TARGET_MULT   = 0.99     # 2차 익절 = SMA200 × 이 값. 200선은 저항이라 정확히
                             # 닿기 전에 밀리는 경우가 많아 살짝 아래에서 받는다
T2_MIN_GAIN_PCT   = 0.02     # 단, 2차 목표는 진입가 +2% 아래로는 못 내려간다.
                             # 진입가가 200선 바로 아래(-1~2%)면 SMA200×0.98 이
                             # 진입가보다 낮아져 즉시 손실 청산되기 때문
MAX_HOLD_BARS     = 60       # 시간 청산 (선별된 셋업은 길게 끌수록 유리)
# 청산 방식:
#   'full'      = 손절 + 1차 피보 익절 + SMA200 익절 + 시간청산 (기본)
#   'stop_time' = 손절 + 시간청산만. 익절 목표 없이 정해진 기간만 들고 간다
EXIT_MODE         = 'full'
SLIPPAGE_PCT      = 0.001    # 편도 슬리피지/수수료

# ── 회복 검증 후보 지표 ──
RSI_PERIOD        = 14
DIV_LOOKBACK      = 60       # RSI 다이버전스 탐색 구간
DIV_WINDOW        = 3        # 로컬 저점 판정 창
VOL_MA            = 20       # 거래량/거래대금 평균 기간

# ── 추가 진입 필터 ──
# filter_lab.py 로 18개 후보를 전/후반 분할 검증해 살아남은 것만 켜둔다.
# 거래량 급증·RSI 수준·단기 이평 정배열은 전부 기대값을 떨어뜨려 채택하지 않았다.
USE_ROOM_FILTER   = True     # SMA200 까지 여유가 있어야 2차 목표에서 먹을 게 있다
MIN_SMA200_ROOM   = 8.0      # 종가 대비 SMA200 이 최소 몇 % 위에 있어야 하는가
USE_ATR_FILTER    = True     # 과도한 변동성 종목 제외
MAX_ATR_PCT       = 4.0      # ATR(14) / 종가
USE_RSI_DIV_FILTER = False   # 효과는 있으나 통과율 13%로 표본이 얇아 기본 off

# 2차 실험(홀드아웃 검증 통과)에서 채택한 3종. 전반기(~2016)만 보고 고른 뒤
# 후반기(2017~)를 손대지 않고 확인했을 때 기대값 +0.370R → +0.353R 로 거의
# 그대로 유지됐다. 셋을 AND 로 걸면 기대값 +0.14R → +0.36R, PF 1.26 → 1.68.
USE_DRYUP_FILTER  = True     # 눌림 구간에 거래량이 말라야 한다 (투매가 끝난 신호)
MAX_VOL_DRYUP     = 1.0      # 최근 5일 평균 거래량 / 20일 평균
USE_FIBNEAR_FILTER = True    # 피보나치 레벨에 붙어 있어야 한다 (지지 확인)
MAX_FIB_NEAR_PCT  = 2.0      # 가장 가까운 레벨까지 거리(%)
USE_CLOSE_GT_MA5  = True     # 종가가 5일선 위 (단기 반전 확인)


# ============================================================
# 데이터 로딩
# ============================================================
_COL_MAP = {
    'date': 'Date', 'datetime': 'Date', '날짜': 'Date',
    'open': 'Open', '시가': 'Open',
    'high': 'High', '고가': 'High',
    'low': 'Low', '저가': 'Low',
    'close': 'Close', '종가': 'Close', 'adj close': 'Close',
    'volume': 'Volume', '거래량': 'Volume',
}


def normalize_ohlcv(df):
    """컬럼명이 한글/영문 어느 쪽이든 Date/Open/High/Low/Close/Volume 로 통일."""
    ren = {}
    for c in df.columns:
        key = str(c).strip().lstrip('﻿').lower()
        if key in _COL_MAP:
            ren[c] = _COL_MAP[key]
    df = df.rename(columns=ren)
    need = ['Date', 'Open', 'High', 'Low', 'Close']
    if any(c not in df.columns for c in need):
        return None
    if 'Volume' not in df.columns:
        df['Volume'] = np.nan

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True).dt.tz_localize(None)
    for c in ('Open', 'High', 'Low', 'Close', 'Volume'):
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close'])
    df = df[df['Close'] > 0]
    df = df.sort_values('Date').drop_duplicates('Date', keep='last').reset_index(drop=True)
    return df if len(df) else None


# 데이터 위생 파라미터
SPIKE_RATIO = 2.5   # 1봉만에 ±150% 튀었다가 곧바로 되돌리면 오류틱으로 간주
LEVEL_SHIFT = 5.0   # 되돌리지 않는 5배 이상 점프 = 티커 재사용 / 미조정 병합


def clean_ohlcv(df, spike=SPIKE_RATIO, shift=LEVEL_SHIFT):
    """
    수집 원본에 섞인 오류틱을 걷어낸다. 이걸 안 하면 단발 이상치 하나가
    SMA200 터치 청산에 걸려 수익률 수천 %짜리 가짜 트레이드를 만든다.
      1) 다음 봉에서 곧바로 되돌리는 단발 스파이크 봉 → 제거
      2) 되돌리지 않는 shift배 이상 레벨 점프 → 그 이전 구간 전부 절단
      3) 종가/시가 대비 말이 안 되는 고가·저가 → 클램프
    반환: (df, 제거봉수, 절단봉수)
    """
    c = df['Close'].values.astype(float)
    n = len(c)
    if n < 5:
        return df, 0, 0

    r = np.ones(n); r[1:] = c[1:] / c[:-1]
    dev = np.maximum(r, 1.0 / r)
    two = np.ones(n); two[1:-1] = c[2:] / c[:-2]      # 앞뒤 봉 사이 순변화
    dev2 = np.maximum(two, 1.0 / two)

    spike_mask = np.zeros(n, bool)
    spike_mask[1:-1] = (dev[1:-1] > spike) & (dev2[1:-1] < spike)
    n_spike = int(spike_mask.sum())
    d = df[~spike_mask].reset_index(drop=True) if n_spike else df

    c2 = d['Close'].values.astype(float)
    n_cut = 0
    if len(c2) > 2:
        r2 = c2[1:] / c2[:-1]
        bad = np.where((r2 > shift) | (r2 < 1.0 / shift))[0]
        if bad.size:
            n_cut = int(bad[-1]) + 1
            d = d.iloc[n_cut:].reset_index(drop=True)

    if len(d) < 5:
        return d, n_spike, n_cut

    o = d['Open'].values.astype(float); cl = d['Close'].values.astype(float)
    pc = np.roll(cl, 1); pc[0] = o[0]
    hi_cap = np.maximum.reduce([o, cl, pc]) * spike
    lo_flr = np.minimum.reduce([o, cl, pc]) / spike
    d['High'] = np.maximum(np.minimum(d['High'].values, hi_cap), np.maximum(o, cl))
    d['Low'] = np.minimum(np.maximum(d['Low'].values, lo_flr), np.minimum(o, cl))
    return d, n_spike, n_cut


def load_csv(path, clean=True):
    """단일 CSV → 정규화된 OHLCV DataFrame (실패 시 None)."""
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except Exception:
        try:
            df = pd.read_csv(path, encoding='cp949')
        except Exception:
            return None
    df = normalize_ohlcv(df)
    if df is None or not clean:
        return df
    df, _, _ = clean_ohlcv(df)
    return df if len(df) else None


def collect_csv_files(data_root, groups=None, coin_daily_only=True):
    """
    데이터 루트 아래 폴더들을 훑어 (symbol, group, path) 목록을 만든다.
    groups=None 이면 기본 3개 그룹(us_stock / index / coin) 전부.
    """
    spec = {
        'us_stock': 'test_us_stock_data_1000_20',
        'index':    'index_chart',
        'coin':     'coin',
        'kr_stock': 'test_kr_stock_data_500_20',
        'us_week':  'test_us_stock_week_data_3000_20',
    }
    if groups is None:
        groups = ['us_stock', 'index', 'coin']

    out = []
    for g in groups:
        folder = os.path.join(data_root, spec.get(g, g))
        if not os.path.isdir(folder):
            print(f"  [WARN] 폴더 없음: {folder}")
            continue
        for p in sorted(glob.glob(os.path.join(folder, '*.csv'))):
            sym = os.path.splitext(os.path.basename(p))[0]
            if g == 'coin':
                if 'DVOL' in sym.upper():          # 변동성 지수 — 가격 차트 아님
                    continue
                if coin_daily_only and not sym.lower().endswith('daily'):
                    continue
            out.append((sym, g, p))
    return out


# ============================================================
# 지표
# ============================================================
def sma(values, period):
    return pd.Series(values).rolling(period, min_periods=period).mean().values


def _rolling_mad(values, period):
    """이동 평균절대편차 — CCI 분모. sliding_window_view 로 벡터화."""
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape[0], np.nan)
    if x.shape[0] < period:
        return out
    w = sliding_window_view(x, period)
    out[period - 1:] = np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1)
    return out


def cci(high, low, close, period=CCI_PERIOD):
    tp = (np.asarray(high, float) + np.asarray(low, float) + np.asarray(close, float)) / 3.0
    ma = pd.Series(tp).rolling(period, min_periods=period).mean().values
    md = _rolling_mad(tp, period)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = (tp - ma) / (0.015 * md)
    out[~np.isfinite(out)] = np.nan
    return out


def atr(high, low, close, period=ATR_PERIOD):
    h = np.asarray(high, float); l = np.asarray(low, float); c = np.asarray(close, float)
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values


def fib_bounds(dates, high, low, years=FIB_YEARS, min_obs=FIB_MIN_OBS):
    """
    각 시점 기준 '최대 N년치' 구간의 고점/저점을 시간 기반 롤링으로 산출.
    데이터가 N년보다 짧으면 있는 만큼만 사용(=최대 N년).
    """
    idx = pd.DatetimeIndex(dates)
    win = '%dD' % int(years * 365.25)
    hi = pd.Series(np.asarray(high, float), index=idx).rolling(win, min_periods=min_obs).max().values
    lo = pd.Series(np.asarray(low, float), index=idx).rolling(win, min_periods=min_obs).min().values
    return hi, lo


def rsi(close, period=RSI_PERIOD):
    """RSI — scan_daily.py 의 _calc_rsi 와 동일 정의(단순이동평균 방식)."""
    s = pd.Series(np.asarray(close, float))
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).values


def rsi_bullish_divergence(close, rsi_vals, i,
                           lookback=DIV_LOOKBACK, window=DIV_WINDOW):
    """
    i번째 봉 기준 RSI 상승 다이버전스 판정.
    최근 lookback 구간의 로컬 저점 2개를 찾아 가격은 lower low,
    RSI는 higher low 인지 본다. (scan_daily.py 의 정의를 그대로 옮김)
    전 구간이 아니라 진입 후보 봉에서만 호출하는 전제로 루프를 쓴다.
    """
    start = max(0, i - lookback + 1)
    pr = np.asarray(close, float)[start:i + 1]
    rv = np.asarray(rsi_vals, float)[start:i + 1]
    n = len(pr)
    if n < window * 2 + 2:
        return False

    minima = []
    for k in range(window, n - window):
        if pr[k] <= pr[k - window:k + window + 1].min() and np.isfinite(rv[k]):
            minima.append((pr[k], rv[k]))
    if len(minima) < 2:
        return False
    p1, r1 = minima[-2]
    p2, r2 = minima[-1]
    return bool(p2 < p1 * 0.995 and r2 > r1 + 1.0)


def swing_anchors(dates, high, low, pct=None, max_years=None, min_leg=None):
    """
    최근 상승 파동(의미있는 저점 → 그 뒤 최고가)을 각 봉의 피보나치 앵커로 삼는다.

    · 저점은 ZigZag 로 확정한다 — 저점에서 pct 만큼 반등해야 비로소 저점으로 인정.
    · 고점은 **확정 피벗이 아니라 그 저점 이후의 최고가**를 쓴다. 확정 피벗만 쓰면
      신고가를 계속 내는 종목이 몇 년 전 파동에 묶여 되돌림이 음수로 나온다.
    · 둘 다 i 시점까지의 데이터만 쓰므로 미래를 보지 않는다.

    반환: (anchor_high, anchor_low) — 유효한 파동이 없으면 NaN
    """
    pct = SWING_PCT if pct is None else pct
    max_years = SWING_MAX_YEARS if max_years is None else max_years
    min_leg = SWING_MIN_LEG_PCT if min_leg is None else min_leg

    h = np.asarray(high, float); l = np.asarray(low, float)
    n = len(h)
    a_hi = np.full(n, np.nan); a_lo = np.full(n, np.nan)
    if n < 3:
        return a_hi, a_lo
    dt = pd.DatetimeIndex(dates)

    up = False                     # 저점을 찾는 중부터 시작
    ext_p, ext_i = l[0], 0         # 추적 중인 극단값(하락 중이면 저점 후보)
    lo_p, lo_i = np.nan, -1        # 확정된 저점
    run_hi = np.nan                # 확정 저점 이후의 최고가

    for i in range(1, n):
        if up:
            # 상승 국면: 최고가 갱신, pct 만큼 밀리면 다시 저점 탐색 국면으로
            if h[i] > ext_p:
                ext_p, ext_i = h[i], i
            elif l[i] <= ext_p * (1.0 - pct):
                up = False
                ext_p, ext_i = l[i], i
        else:
            if l[i] < ext_p:
                ext_p, ext_i = l[i], i
            elif h[i] >= ext_p * (1.0 + pct):
                lo_p, lo_i = ext_p, ext_i          # 저점 확정
                run_hi = np.nanmax(h[lo_i:i + 1])  # 그 저점 이후 최고가
                up = True
                ext_p, ext_i = h[i], i

        if lo_i >= 0:
            if h[i] > run_hi:
                run_hi = h[i]
            if (run_hi > lo_p * (1.0 + min_leg)
                    and (dt[i] - dt[lo_i]).days <= max_years * 365.25):
                a_hi[i], a_lo[i] = run_hi, lo_p
    return a_hi, a_lo


def fib_level(low, high, ratio):
    """고점=0.0 / 저점=1.0 기준 되돌림 가격. 비율이 클수록 낮은 가격."""
    return high - ratio * (high - low)


def fib_levels(low, high, ratios=FIB_RATIOS):
    return {r: fib_level(low, high, r) for r in ratios}


def fib_position(price, low, high):
    """고점에서 얼마나 되돌렸는지. 0 = 고점, 1 = 저점."""
    rng = high - low
    return float('nan') if rng <= 0 else (high - price) / rng


# ============================================================
# 신호 계산
# ============================================================
def compute_indicators(df):
    """
    OHLCV DataFrame 에 지표/조건 컬럼을 붙여 반환.
    붙는 컬럼: SMA200, CCI, ATR, FIB_HI, FIB_LO, FIB_FLOOR, FIB_CEIL, FIB_POS,
              CROSS_UP, SIGNAL_BAR, ENTRY_OK
    (FIB_POS = 되돌림 깊이. 0 = 10년 고점, 1 = 10년 저점)
    """
    d = df.copy()
    h, l, c = d['High'].values, d['Low'].values, d['Close'].values

    d['SMA200'] = sma(c, SMA_PERIOD)
    d['CCI']    = cci(h, l, c, CCI_PERIOD)
    d['ATR']    = atr(h, l, c, ATR_PERIOD)

    # 회복 검증 후보 지표
    v = d['Volume'].values
    d['RSI']    = rsi(c, RSI_PERIOD)
    d['SMA5']   = sma(c, 5)
    d['SMA10']  = sma(c, 10)
    d['SMA20']  = sma(c, 20)
    d['VOL_MA'] = pd.Series(v).rolling(VOL_MA, min_periods=VOL_MA).mean().values
    d['VOL_MA5'] = pd.Series(v).rolling(5, min_periods=5).mean().values
    d['DVOL']   = pd.Series(c * v).rolling(VOL_MA, min_periods=VOL_MA).mean().values
    d['SMA50']  = sma(c, 50)

    # 추세 구조 / 위치
    cs = pd.Series(c)
    s200s = pd.Series(d['SMA200'].values)
    with np.errstate(divide='ignore', invalid='ignore'):
        d['SMA200_SLOPE'] = (s200s / s200s.shift(20) - 1.0).values * 100.0
        d['HIGH52_DIST'] = (cs / cs.rolling(252, min_periods=120).max() - 1.0).values * 100.0
        d['LOW52_DIST'] = (cs / cs.rolling(252, min_periods=120).min() - 1.0).values * 100.0
        bb_ma = cs.rolling(20, min_periods=20).mean()
        bb_sd = cs.rolling(20, min_periods=20).std()
        d['BB_PCTB'] = ((cs - (bb_ma - 2 * bb_sd)) / (4 * bb_sd)).values
    # 200선 아래에 머문 연속 봉수
    below = (c < d['SMA200'].values)
    cnt = np.zeros(len(d), dtype=float)
    run = 0
    for k in range(len(d)):
        run = run + 1 if below[k] else 0
        cnt[k] = run
    d['DAYS_BELOW200'] = cnt

    if FIB_MODE == 'swing':
        fhi, flo = swing_anchors(d['Date'].values, h, l)
    else:
        fhi, flo = fib_bounds(d['Date'].values, h, l)
    d['FIB_HI'] = fhi
    d['FIB_LO'] = flo
    # FIB_FLOOR = 허용 최저가(가장 깊은 되돌림), FIB_CEIL = 허용 최고가(가장 얕은 되돌림)
    d['FIB_FLOOR'] = fhi - FIB_MAX_RETRACE * (fhi - flo)
    d['FIB_CEIL'] = fhi - FIB_MIN_RETRACE * (fhi - flo)
    with np.errstate(divide='ignore', invalid='ignore'):
        d['FIB_POS'] = (fhi - c) / (fhi - flo)      # 되돌림 깊이

    cs = pd.Series(d['CCI'].values)
    prev = cs.shift(1)
    # -100 상향 돌파
    cross_up = (cs >= CCI_RECOVER) & (prev < CCI_RECOVER)
    # 돌파 직전 DIP_MAX_BARS 봉 안에 -200 이하 눌림이 있었는지
    dip_ok = cs.rolling(DIP_MAX_BARS, min_periods=1).min().shift(1) <= CCI_OVERSOLD
    valid_cross = (cross_up & dip_ok).fillna(False)
    d['CROSS_UP'] = valid_cross.values

    # 최근 RECOVER_WINDOW 봉 안에 유효 회복이 있었는가 + 그 봉 위치
    vc = valid_cross.astype(float)
    d['SIGNAL_BAR'] = (vc.rolling(RECOVER_WINDOW, min_periods=1).max() > 0).values
    pos = pd.Series(np.where(valid_cross.values, np.arange(len(d)), np.nan))
    d['CROSS_IDX'] = pos.ffill().values

    enough = np.arange(len(d)) >= MIN_BARS - 1
    entry = (
        enough
        & np.isfinite(d['SMA200'].values) & (c < d['SMA200'].values)
        & d['SIGNAL_BAR'].values
        & np.isfinite(d['FIB_FLOOR'].values) & (c >= d['FIB_FLOOR'].values)
        & (c <= d['FIB_CEIL'].values)
        & (d['FIB_HI'].values > d['FIB_LO'].values)
    )

    # ── 추가 진입 필터 ──
    with np.errstate(divide='ignore', invalid='ignore'):
        room = (d['SMA200'].values / c - 1.0) * 100.0
        atr_pct = d['ATR'].values / c * 100.0
    d['SMA200_ROOM'] = room
    d['ATR_PCT'] = atr_pct

    # 가장 가까운 피보나치 레벨까지 거리(%) — 지지 확인용
    near = np.full(len(d), np.inf)
    for r in FIB_RATIOS:
        near = np.minimum(near, np.abs(c - (fhi - r * (fhi - flo))))
    with np.errstate(divide='ignore', invalid='ignore'):
        d['FIB_NEAR_PCT'] = near / c * 100.0
        d['VOL_DRYUP'] = d['VOL_MA5'].values / d['VOL_MA'].values

    if USE_ROOM_FILTER:
        entry = entry & (room >= MIN_SMA200_ROOM)
    if USE_ATR_FILTER:
        entry = entry & (atr_pct <= MAX_ATR_PCT)
    if USE_DRYUP_FILTER:
        entry = entry & (d['VOL_DRYUP'].values <= MAX_VOL_DRYUP)
    if USE_FIBNEAR_FILTER:
        entry = entry & (d['FIB_NEAR_PCT'].values <= MAX_FIB_NEAR_PCT)
    if USE_CLOSE_GT_MA5:
        entry = entry & (c > d['SMA5'].values)
    if USE_RSI_DIV_FILTER:
        # 루프가 필요한 판정이라 후보로 남은 봉에서만 계산한다
        cl, rv = d['Close'].values, d['RSI'].values
        for k in np.where(entry)[0]:
            if not rsi_bullish_divergence(cl, rv, int(k)):
                entry[k] = False

    d['ENTRY_OK'] = entry
    return d


def load_market_context(data_root, symbol='SPY'):
    """
    시장 국면 판단용 지수 시계열. 이 전략은 하락장에서 크게 깨지므로
    '시장 자체가 어떤 상태였나'를 진입 시점 정보로 붙여 검증한다.
    반환: Date 인덱스 DataFrame (MKT_ABOVE200, MKT_SLOPE, MKT_DD)
    """
    import os
    path = os.path.join(data_root, 'index_chart', f'{symbol}.csv')
    df = load_csv(path)
    if df is None:
        return None
    c = df['Close'].values
    s200 = sma(c, SMA_PERIOD)
    prev = pd.Series(s200).shift(20).values
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = (s200 / prev - 1.0) * 100.0
        dd = (c / pd.Series(c).rolling(252, min_periods=60).max().values - 1.0) * 100.0
    return pd.DataFrame({
        'MKT_ABOVE200': c > s200,
        'MKT_SLOPE': slope,
        'MKT_DD': dd,
    }, index=pd.DatetimeIndex(df['Date'].values))


def _fib_support(close, low_n, f_lo, f_hi):
    """
    피보나치 되돌림 '지지 확인' 관련 지표.
      near_pct : 종가에서 가장 가까운 레벨까지 거리(%) — 작을수록 레벨에 붙어 있음
      reclaim  : 최근 저점이 레벨을 뚫었다가 종가가 그 위로 회복 = 지지 확인
    """
    lv = list(fib_levels(f_lo, f_hi).values())
    if not lv or not np.isfinite(close) or close <= 0:
        return np.nan, False
    near = min(abs(close - v) for v in lv) / close * 100.0
    reclaim = any((low_n < v <= close) for v in lv)
    return near, bool(reclaim)


def entry_features(d, i):
    """
    i번째 봉(진입 후보) 시점의 '회복세 검증' 후보 지표들.
    여기서는 판정만 하고 거르지 않는다 — filter_lab.py 가 효과를 측정한다.
    """
    g = lambda k: float(d[k].values[i])
    c = g('Close')
    ma5, ma10, ma20 = g('SMA5'), g('SMA10'), g('SMA20')
    volma = g('VOL_MA')
    vol = float(d['Volume'].values[i])
    lo10 = float(d['Low'].values[max(0, i - SWING_LOOKBACK + 1): i + 1].min())

    return {
        # 유동성 — 과거 시점 시가총액은 구할 수 없어 거래대금으로 대신한다
        'dvol20': g('DVOL'),
        # 회복봉에 거래량이 실렸는가
        'vol_ratio': (vol / volma) if (np.isfinite(volma) and volma > 0) else np.nan,
        # RSI 수준 / 상승 다이버전스(저점 상승)
        'rsi': g('RSI'),
        'rsi_div': rsi_bullish_divergence(d['Close'].values, d['RSI'].values, i),
        # 단기 이평 회복
        'ma5_gt_ma10': bool(ma5 > ma10),
        'ma_align3': bool(ma5 > ma10 > ma20),
        'close_gt_ma5': bool(c > ma5),
        # 변동성 / 반등폭 / 목표까지 여유
        'atr_pct': g('ATR') / c * 100 if c else np.nan,
        'bounce_pct': (c / lo10 - 1) * 100 if lo10 > 0 else np.nan,
        'sma200_room': (g('SMA200') / c - 1) * 100,
        # ── 2차 후보 ──
        # 피보나치 지지 확인
        'fib_near_pct': _fib_support(c, lo10, g('FIB_LO'), g('FIB_HI'))[0],
        'fib_reclaim': _fib_support(c, lo10, g('FIB_LO'), g('FIB_HI'))[1],
        # 추세 구조
        'sma200_slope': g('SMA200_SLOPE'),
        'days_below200': g('DAYS_BELOW200'),
        'high52_dist': g('HIGH52_DIST'),
        'low52_dist': g('LOW52_DIST'),
        'bb_pctb': g('BB_PCTB'),
        'sma50_room': (g('SMA50') / c - 1) * 100,
        # 눌림의 깊이 / 거래량 마름
        'cci_min': float(np.nanmin(
            d['CCI'].values[max(0, i - (RECOVER_WINDOW + DIV_WINDOW * 2)):i + 1])),
        'vol_dryup': (g('VOL_MA5') / g('VOL_MA')
                      if np.isfinite(g('VOL_MA')) and g('VOL_MA') > 0 else np.nan),
        # 저점 구조: 최근 저점이 그 이전 저점보다 높은가(higher low)
        'higher_low': bool(lo10 > float(d['Low'].values[max(0, i - 25):max(1, i - 10)].min())),
    }


def build_trade_plan(entry_close, swing_low, f_lo, f_hi, slippage=SLIPPAGE_PCT):
    """
    진입 종가 기준 손절/1차목표 산출. 실패(위험폭 무효) 시 None.
    반환: entry / stop / t1 / risk / risk_pct / t1_src
    """
    entry = entry_close * (1.0 + slippage)

    stop = swing_low * (1.0 - SWING_BUFFER)
    stop = max(stop, entry * (1.0 - MAX_STOP_PCT))   # 너무 먼 손절 → 12%로 제한
    stop = min(stop, entry * (1.0 - MIN_STOP_PCT))   # 너무 가까운 손절 → 3%로 제한
    risk = entry - stop
    if risk <= 0:
        return None

    t1_src = 'fib'
    cands = [v for v in fib_levels(f_lo, f_hi).values() if v > entry * (1.0 + T1_MIN_GAP_PCT)]
    if cands:
        t1 = min(cands)
    else:
        t1 = entry + T1_FALLBACK_R * risk
        t1_src = 'R'
    if t1 < entry + T1_MIN_R * risk:
        t1 = entry + T1_MIN_R * risk
        t1_src = 'R(min)'
    if t1 > entry + T1_MAX_R * risk:
        t1 = entry + T1_MAX_R * risk
        t1_src = 'R(cap)'

    return {'entry': entry, 'stop': stop, 't1': t1,
            'risk': risk, 'risk_pct': risk / entry, 't1_src': t1_src}


def plan_at(d, i):
    """지표가 계산된 DataFrame 의 i번째 봉에서 진입한다고 가정한 청산 플랜."""
    lo_win = d['Low'].values[max(0, i - SWING_LOOKBACK + 1): i + 1]
    if lo_win.size == 0:
        return None
    return build_trade_plan(float(d['Close'].values[i]), float(lo_win.min()),
                            float(d['FIB_LO'].values[i]), float(d['FIB_HI'].values[i]))
