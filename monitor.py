"""
CCI 반등 + 200일선 하단 + 피보나치 되돌림 — 모니터링 + Vercel 배포
============================================================
로컬 KR/US 일봉 CSV 장종료 갱신 후 자동 스캔 -> HTML 리포트 -> Git push -> Vercel 배포

[사용법]
  python monitor.py                        # KR/US 로컬 CSV 원샷 스캔
  python monitor.py --market kr            # 한국만
  python monitor.py --market us            # 미국만
  python monitor.py --daemon               # 24시간 스케줄러
  python monitor.py --after-update kr      # update_charts.py가 호출하는 내부 옵션
  python monitor.py --no-git               # Git push 안 함
  python monitor.py --strict               # 백테스트 동일 조건(추가 필터 전부 켜기)
  python monitor.py --no-git               # Git push 없이 결과 생성

[산출물]
  public/index.html       <- Vercel 에서 서빙하는 메인 페이지
  public/watchlist.txt    <- TradingView 워치리스트
"""

import os
import sys
import time
import json
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import strategy as ST

# ── 모니터 전용: 스캔 범위를 넓히기 위해 백테스트보다 느슨한 조건 ──
# --strict 플래그로 백테스트 동일 조건 복원 가능
_ORIG_RECOVER_WINDOW = ST.RECOVER_WINDOW
ST.RECOVER_WINDOW     = 2
ST.USE_ROOM_FILTER    = False
ST.USE_ATR_FILTER     = False
ST.USE_DRYUP_FILTER   = False
ST.USE_FIBNEAR_FILTER = False
ST.USE_CLOSE_GT_MA5   = False
ST.USE_RSI_DIV_FILTER = False

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'public')
STATE_DIR  = os.path.join(BASE_DIR, 'monitor_state')
BACKTEST_DIR = Path(os.environ.get(
    'CCI_BACKTEST_DIR', r'C:\Users\user\Documents\claude_coding\backtest'))
DAILY_DIRS = {
    'kr': BACKTEST_DIR / 'kr_stock' / 'daily',
    'us': BACKTEST_DIR / 'us_stock' / 'daily',
}
VIEW_BARS    = 200
MAX_CHARTS_SIGNAL = 30
MAX_CHARTS_WATCH  = 10

UP_C, DN_C = '#26a69a', '#ef5350'

EXCH_MAP = {
    'NASDAQ': 'NASDAQ', 'NYSE': 'NYSE', 'S&P500': 'NASDAQ',
    'KOSPI': 'KRX', 'KOSDAQ': 'KRX', 'KRX': 'KRX',
}

# 스케줄 (KST 기준)
KR_SCAN_TIME     = (15, 40)
US_SCAN_TIME     = (6, 10)
US_SCAN_DAYS     = {1, 2, 3, 4, 5}

ENABLE_GIT_PUSH = True


# ============================================================
# 로컬 CSV 종목 목록
# ============================================================
def local_instruments(market, limit=0):
    """백테스트와 동일한 일봉 CSV만 사용한다. 네트워크 수신은 하지 않는다."""
    folder = DAILY_DIRS[market]
    if not folder.is_dir():
        raise FileNotFoundError(f'일봉 폴더 없음: {folder}')
    instruments = []
    for path in sorted(folder.glob('*.csv')):
        stem = path.stem
        if market == 'kr':
            name, sep, code = stem.rpartition('_')
            if not sep or not code.isdigit() or len(code) != 6:
                continue
            instruments.append((code, {'name': name, 'market': 'KRX'}, path))
        else:
            instruments.append((stem, {'name': stem, 'market': 'NASDAQ'}, path))
    if limit:
        instruments = instruments[:limit]
    print(f'  로컬 {market.upper()} 일봉 CSV {len(instruments)}개: {folder}')
    return instruments


# ============================================================
# 스캔
# ============================================================
def evaluate(sym, info, df):
    d = ST.compute_indicators(df)
    i = len(d) - 1
    c = float(d['Close'].values[i])
    s200 = d['SMA200'].values[i]
    fflr = d['FIB_FLOOR'].values[i]
    fceil = d['FIB_CEIL'].values[i]
    f_hi = d['FIB_HI'].values[i]
    f_lo = d['FIB_LO'].values[i]
    cci_now = d['CCI'].values[i]

    if not (np.isfinite(s200) and np.isfinite(fflr)
            and np.isfinite(cci_now) and f_hi > f_lo):
        return None
    if c >= s200:
        return None
    if c < fflr or c > fceil:
        return None

    tier = None
    if bool(d['ENTRY_OK'].values[i]):
        tier = 'signal'
    elif cci_now <= ST.CCI_RECOVER:
        tier = 'watch'
    if tier is None:
        return None

    ci = d['CROSS_IDX'].values[i]
    cross_i = int(ci) if np.isfinite(ci) else None
    if tier == 'watch':
        cross_i = None

    room = float((s200 / c - 1) * 100) if c > 0 else 0.0

    return {
        'symbol': sym, 'name': info.get('name', sym),
        'market': info.get('market', ''),
        'tier': tier, 'df': d, 'idx': i,
        'date': pd.Timestamp(d['Date'].values[i]).strftime('%Y-%m-%d'),
        'close': c, 'sma200': float(s200),
        'sma_gap': (c / s200 - 1) * 100,
        'cci': float(cci_now),
        'cci_dip': float(np.nanmin(
            d['CCI'].values[max(0, i - ST.RECOVER_WINDOW - ST.DIP_MAX_BARS):i + 1])),
        'cross_i': cross_i,
        'cross_date': (pd.Timestamp(d['Date'].values[cross_i]).strftime('%Y-%m-%d')
                       if cross_i is not None else '-'),
        'room': room,
        'fib_hi': float(f_hi), 'fib_lo': float(f_lo),
        'fib_floor': float(fflr),
        'fib_pos': ST.fib_position(c, float(f_lo), float(f_hi)),
        'rsi': float(d['RSI'].values[i]),
        'rsi_div': ST.rsi_bullish_divergence(d['Close'].values,
                                             d['RSI'].values, i),
    }


# ============================================================
# TradingView 심볼 변환
# ============================================================
def tv_symbol(sym, market):
    ex = EXCH_MAP.get(market, 'NASDAQ')
    clean = sym.split('.')[0]
    return f'{ex}:{clean}'


# ============================================================
# 차트
# ============================================================
def build_chart(rec, div_id):
    d = rec['df']
    i = rec['idx']
    f_lo, f_hi = rec['fib_lo'], rec['fib_hi']
    levels = ST.fib_levels(f_lo, f_hi)
    v = d.iloc[max(0, i - VIEW_BARS + 1):i + 1]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=('일봉 · SMA200 · 피보나치',
                        f'CCI({ST.CCI_PERIOD})'))

    fig.add_trace(go.Candlestick(
        x=v['Date'], open=v['Open'], high=v['High'],
        low=v['Low'], close=v['Close'],
        increasing_line_color=UP_C, decreasing_line_color=DN_C,
        increasing_fillcolor=UP_C, decreasing_fillcolor=DN_C,
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=v['Date'], y=v['SMA200'], mode='lines',
        line=dict(color='#ffb74d', width=1.5), showlegend=False,
        hovertemplate='SMA200 %{y:.2f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[d['Date'].values[i]], y=[rec['close']],
        mode='markers',
        marker=dict(symbol='circle', size=8, color='#ffd54f',
                    line=dict(color='#000', width=1)),
        showlegend=False,
        hovertemplate='종가 %{y:.2f}<extra></extra>'), row=1, col=1)
    if rec['cross_i'] is not None and rec['cross_i'] >= v.index[0]:
        k = rec['cross_i']
        fig.add_trace(go.Scatter(
            x=[d['Date'].values[k]], y=[d['Low'].values[k] * 0.97],
            mode='markers',
            marker=dict(symbol='triangle-up', size=11, color='#4fc3f7'),
            showlegend=False,
            hovertemplate='CCI 회복<extra></extra>'), row=1, col=1)

    lo_c = float(v['Low'].min())
    hi_c = max(float(v['High'].max()),
               float(np.nanmax(v['SMA200'].values)))
    span = max(hi_c - lo_c, 1e-9)
    ylo = min(lo_c * 0.98,
              max(levels.get(ST.FIB_MAX_RETRACE, lo_c), lo_c - 0.5 * span))
    yhi = hi_c * 1.03
    for r, px in levels.items():
        if not (ylo <= px <= yhi):
            continue
        key = (r == ST.FIB_MAX_RETRACE)
        fig.add_hline(
            y=px, row=1, col=1,
            line=dict(color='#ff9800' if key else '#5b6478',
                      width=1.4 if key else 0.8,
                      dash='solid' if key else 'dot'),
            annotation_text=f'{r:.3f} {px:,.2f}',
            annotation_position='top left',
            annotation_font=dict(size=9,
                                 color='#ff9800' if key else '#8898b0'))
    fig.update_yaxes(range=[ylo, yhi], row=1, col=1)

    fig.add_trace(go.Scatter(
        x=v['Date'], y=v['CCI'], mode='lines',
        line=dict(color='#ba68c8', width=1.2), showlegend=False,
        hovertemplate='CCI %{y:.0f}<extra></extra>'), row=2, col=1)
    for y0, col, dash in [(0, '#5b6478', 'dot'),
                          (ST.CCI_RECOVER, '#4fc3f7', 'dash'),
                          (ST.CCI_OVERSOLD, '#ef5350', 'dash')]:
        fig.add_hline(y=y0, row=2, col=1,
                      line=dict(color=col, width=1, dash=dash),
                      annotation_text=f'{y0:.0f}',
                      annotation_position='left',
                      annotation_font=dict(size=9, color=col))
    cmin = float(np.nanmin(v['CCI'].values))
    cmax = float(np.nanmax(v['CCI'].values))
    fig.update_yaxes(
        range=[min(cmin, -250) - 20, max(cmax, 120) + 20], row=2, col=1)

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(rangebreaks=[dict(bounds=['sat', 'mon'])])
    fig.update_layout(
        height=440, margin=dict(l=8, r=8, t=26, b=8),
        paper_bgcolor='#161b22', plot_bgcolor='#161b22',
        font=dict(color='#c8d2e0', size=10),
        hovermode='x unified', dragmode='pan')
    fig.update_xaxes(gridcolor='#232a36', zeroline=False)
    fig.update_yaxes(gridcolor='#232a36', zeroline=False, side='right')
    for a in fig.layout.annotations[:2]:
        a.font.size = 11
        a.font.color = '#8898b0'

    return fig.to_html(include_plotlyjs=False, full_html=False,
                       div_id=div_id,
                       config={'displayModeBar': False, 'scrollZoom': True})


def render_card(rec, div_id):
    tv = tv_symbol(rec['symbol'], rec['market'])
    tv_link = f'https://www.tradingview.com/chart/?symbol={tv}'
    esc = lambda s: (str(s).replace('&', '&amp;')
                     .replace('<', '&lt;').replace('>', '&gt;'))
    meta = [
        f'200선여유 <b>{rec["room"]:+.1f}%</b>',
        f'CCI <b>{rec["cci"]:.0f}</b>',
        f'RSI <b>{rec["rsi"]:.0f}</b>'
        + (' <b style="color:#ffd54f">다이버전스</b>'
           if rec['rsi_div'] else ''),
        f'되돌림 <b>{rec["fib_pos"]:.3f}</b>',
    ]
    if rec['cross_date'] != '-':
        meta.insert(1, f'회복일 <b>{rec["cross_date"]}</b>')

    chart_html = build_chart(rec, div_id)
    return (
        f'<div class="card"><div class="card-hd">'
        f'<span class="tkr">{esc(rec["symbol"])}</span>'
        f'<span class="cname">{esc(rec["name"])[:35]}</span>'
        f'<span class="px">{rec["close"]:,.2f}</span>'
        f'<a class="tv" href="{tv_link}" target="_blank" '
        f'rel="noopener noreferrer">TV</a></div>'
        f'<div class="card-mt">'
        f'{"".join(f"<span>{m}</span>" for m in meta)}</div>'
        f'{chart_html}</div>')


# ============================================================
# HTML
# ============================================================
CSS = """\
:root{--bg:#0e1117;--card:#161b22;--line:#2a3040;--fg:#e4e8f0;--dim:#8898b0;}
*{box-sizing:border-box}
body{font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif;
  background:var(--bg);color:var(--fg);margin:0;min-height:100vh;}
header{padding:16px 20px;border-bottom:1px solid var(--line);
  position:sticky;top:0;background:var(--bg);z-index:20;}
h1{font-size:18px;margin:0;display:inline-block;}
.sub{color:var(--dim);font-size:12px;margin-top:5px;line-height:1.6;}
nav{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;}
nav a{font-size:12px;color:var(--dim);text-decoration:none;
  border:1px solid var(--line);border-radius:14px;padding:4px 13px;
  transition:all .2s;}
nav a:hover{color:#fff;border-color:#4fc3f7;}
nav a.dl{background:#2962ff;color:#fff;border-color:#2962ff;font-weight:600;
  cursor:pointer;}
nav a.dl:hover{background:#1e4fd0;}
.mkt-section{margin:20px 0;scroll-margin-top:120px;
  border-top:1px solid var(--line);padding-top:16px;}
.mkt-hd{padding:0 20px;margin-bottom:12px;}
.mkt-hd h2{font-size:16px;margin:0 0 4px;}
.mkt-hd .meta{font-size:11px;color:var(--dim);}
.sec-title{font-size:14px;font-weight:700;padding:0 20px;
  margin:16px 0 8px;color:#ff9800;}
.sec-title.watch{color:#4fc3f7;}
.sec-title .cnt{font-size:11px;color:var(--dim);font-weight:500;
  border:1px solid var(--line);border-radius:9px;padding:1px 7px;
  margin-left:4px;}
.rule{margin:0 20px 14px;padding:10px 14px;background:var(--card);
  border:1px solid var(--line);border-radius:7px;font-size:12px;
  color:var(--dim);line-height:1.75;}
.rule b{color:#ffd54f;font-weight:600;}
.tblwrap{margin:0 20px 16px;overflow-x:auto;
  border:1px solid var(--line);border-radius:7px;}
table{border-collapse:collapse;width:100%;font-size:12px;
  background:var(--card);}
th,td{padding:6px 9px;text-align:right;white-space:nowrap;
  border-bottom:1px solid #21262f;}
th{background:#1b2029;color:var(--dim);font-weight:600;
  position:sticky;top:0;font-size:11px;}
td.l,th.l{text-align:left;}
tr:hover td{background:#1c222c;}
.pos{color:#26a69a;} .neg{color:#ef5350;}
.hl{color:#ffd54f;font-weight:600;}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));
  gap:12px;padding:0 20px;}
.card{background:var(--card);border:1px solid var(--line);
  border-radius:7px;overflow:hidden;}
.card-hd{display:flex;align-items:center;gap:8px;padding:7px 11px;
  border-bottom:1px solid var(--line);background:#1b2029;flex-wrap:wrap;}
.card-hd .tkr{font-size:14px;font-weight:700;letter-spacing:.02em;}
.card-hd .cname{font-size:12px;color:var(--dim);overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;max-width:180px;}
.card-hd .px{margin-left:auto;font-size:13px;font-weight:600;color:#cfd8e3;}
.card-hd .tv{font-size:11px;font-weight:600;color:#2962ff;
  text-decoration:none;border:1px solid #2962ff;border-radius:4px;
  padding:2px 7px;}
.card-hd .tv:hover{background:#2962ff;color:#fff;}
.card-mt{display:flex;gap:12px;flex-wrap:wrap;padding:6px 11px;
  font-size:11px;color:var(--dim);border-bottom:1px solid var(--line);}
.card-mt b{color:#cfd8e3;font-weight:600;}
.empty{padding:14px 20px;color:#556070;font-size:13px;}
footer{padding:18px 20px 30px;color:#4a5464;font-size:11px;line-height:1.7;
  border-top:1px solid var(--line);margin-top:20px;}
@media(max-width:600px){
  .grid{grid-template-columns:1fr;}
  .card-hd .cname{max-width:120px;}
  nav{gap:5px;}
}
"""


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def summary_table(recs):
    if not recs:
        return '<div class="empty">해당 없음</div>'
    rows = [
        '<div class="tblwrap"><table><thead><tr>',
        '<th class="l">티커</th><th class="l">종목명</th><th>종가</th>',
        '<th>200선</th><th>200선여유</th><th>CCI</th><th>최저CCI</th>',
        '<th>회복일</th><th>되돌림</th><th>RSI</th><th>다이버전스</th>',
        '</tr></thead><tbody>']
    for r in recs:
        rows.append(
            f'<tr><td class="l hl">{_esc(r["symbol"])}</td>'
            f'<td class="l">{_esc(r["name"])[:30]}</td>'
            f'<td>{r["close"]:,.2f}</td>'
            f'<td>{r["sma200"]:,.2f}</td>'
            f'<td class="pos">{r["room"]:+.1f}%</td>'
            f'<td>{r["cci"]:.0f}</td>'
            f'<td class="neg">{r["cci_dip"]:.0f}</td>'
            f'<td>{_esc(r["cross_date"])}</td>'
            f'<td class="pos">{r["fib_pos"]:.3f}</td>'
            f'<td>{r["rsi"]:.0f}</td>'
            f'<td class="{"hl" if r["rsi_div"] else ""}">'
            f'{"OK" if r["rsi_div"] else "-"}</td></tr>')
    rows.append('</tbody></table></div>')
    return ''.join(rows)


def market_section_html(market_key, market_label, scan_result):
    if scan_result is None:
        return (f'<div class="mkt-section" id="{market_key}">'
                f'<div class="mkt-hd"><h2>{market_label}</h2>'
                f'<div class="meta">아직 스캔하지 않음</div></div></div>')

    sig = scan_result['signals']
    watch = scan_result['watch']
    parts = [
        f'<div class="mkt-section" id="{market_key}">',
        f'<div class="mkt-hd"><h2>{market_label}</h2>',
        f'<div class="meta">스캔: {scan_result["scan_time"]} · '
        f'{scan_result["scanned"]}종목 · 기준일 {scan_result["asof"]}'
        f'</div></div>',
    ]

    parts.append(
        f'<div class="sec-title">진입 신호 '
        f'<span class="cnt">{len(sig)}</span></div>')
    parts.append(summary_table(sig))
    if sig:
        n = min(len(sig), MAX_CHARTS_SIGNAL)
        parts.append('<div class="grid">')
        for k, r in enumerate(sig[:n]):
            parts.append(r.get('_card', ''))
        parts.append('</div>')

    parts.append(
        f'<div class="sec-title watch">관찰 '
        f'<span class="cnt">{len(watch)}</span></div>')
    parts.append(summary_table(watch))
    if watch:
        n = min(len(watch), MAX_CHARTS_WATCH)
        if len(watch) > n:
            parts.append(
                f'<div class="empty">차트는 상위 {n}개만 표시</div>')
        parts.append('<div class="grid">')
        for k, r in enumerate(watch[:n]):
            parts.append(r.get('_card', ''))
        parts.append('</div>')

    parts.append('</div>')
    return ''.join(parts)


def build_html(results, watchlist_content):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    total_sig = sum(len(r['signals']) for r in results.values() if r)
    total_watch = sum(len(r['watch']) for r in results.values() if r)

    def mc(key):
        r = results.get(key)
        return (len(r['signals']), len(r['watch'])) if r else (0, 0)
    kr_s, kr_w = mc('kr')
    us_s, us_w = mc('us')

    wl_escaped = _esc(watchlist_content)

    parts = [
        '<!DOCTYPE html>\n<html lang="ko"><head><meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f'<title>CCI 반등 모니터</title>',
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
        f'<style>{CSS}</style></head><body>',

        '<header><h1>CCI 반등 + 200일선 + 피보나치 모니터</h1>',
        f'<div class="sub">마지막 갱신 {now} · '
        f'진입 신호 <b>{total_sig}</b>건 / 관찰 <b>{total_watch}</b>건</div>',
        '<nav>',
        f'<a href="#kr">한국 ({kr_s}+{kr_w})</a>',
        f'<a href="#us">미국 ({us_s}+{us_w})</a>',
        f'<a class="dl" href="#" id="dl-btn">'
        f'워치리스트 ({total_sig + total_watch})</a>',
        '<a href="archive.html">아카이브</a>',
        '</nav></header>',

        '<div class="rule" style="margin:16px 20px">',
        '<b>진입 조건</b> &nbsp;'
        f'1. 종가 &lt; SMA200 &nbsp;'
        f'2. CCI({ST.CCI_PERIOD}) &le; {ST.CCI_OVERSOLD:.0f} 눌림 후 '
        f'{ST.CCI_RECOVER:.0f} 위로 회복, '
        f'최근 {ST.RECOVER_WINDOW}거래일 내 &nbsp;'
        f'3. 최대 {ST.FIB_YEARS}년 고점(0)·저점(1) 피보나치에서 '
        f'되돌림 {ST.FIB_MIN_RETRACE}~{ST.FIB_MAX_RETRACE} 구간 '
        '(맨 위·맨 아래 제외)<br>',
        '<b>관찰</b> &nbsp;1,3번 충족 + CCI가 아직 -100 아래 (회복 대기 중)',
        '</div>',
    ]

    parts.append(market_section_html(
        'kr', '한국 — 시총 상위 1,000개', results.get('kr')))
    parts.append(market_section_html(
        'us', '미국 — 시총 상위 2,000개', results.get('us')))
    parts.append(
        f'<div id="wl-data" style="display:none">{wl_escaped}</div>')
    parts.append(
        '<script>'
        'document.getElementById("dl-btn").addEventListener("click",'
        'function(e){'
        'e.preventDefault();'
        'var t=document.getElementById("wl-data").textContent;'
        'var b=new Blob([t],{type:"text/plain"});'
        'var a=document.createElement("a");'
        'a.href=URL.createObjectURL(b);'
        'a.download="cci_watchlist.txt";'
        'document.body.appendChild(a);'
        'a.click();'
        'document.body.removeChild(a);'
        '});'
        '</script>')

    parts.append(
        '<footer>'
        '· 장중 실행 시 마지막 봉은 미완성 종가라 장 마감 후 결과와 '
        '달라질 수 있다.<br>'
        '· 투자 판단의 근거로 쓰기 위한 참고 자료이며, '
        '매매 손익은 이용자 책임이다.'
        '</footer></body></html>')
    return ''.join(parts)


# ============================================================
# 워치리스트
# ============================================================
def generate_watchlist(results):
    lines = []
    for mkt in ('kr', 'us'):
        res = results.get(mkt)
        if res is None:
            continue
        for r in res['signals'] + res['watch']:
            lines.append(tv_symbol(r['symbol'], r['market']))
    return '\n'.join(lines)


# ============================================================
# Git push
# ============================================================
def git_push(msg):
    if not ENABLE_GIT_PUSH:
        return
    repo = BASE_DIR
    if not os.path.isdir(os.path.join(repo, '.git')):
        print(f"  [Git] {repo} 는 git 저장소가 아닙니다 -> skip")
        return
    try:
        subprocess.run(['git', '-C', repo, 'add', 'public/'],
                       capture_output=True, timeout=30)
        res = subprocess.run(
            ['git', '-C', repo, 'commit', '-m', msg],
            capture_output=True, encoding='utf-8',
            errors='replace', timeout=30)
        out = (res.stdout or '').strip()
        if 'nothing to commit' in out.lower():
            print(f"  [Git] 변경사항 없음")
            return
        print(f"  [Git] commit: {msg}")
        push = subprocess.run(
            ['git', '-C', repo, 'push'],
            capture_output=True, encoding='utf-8',
            errors='replace', timeout=120)
        if push.returncode == 0:
            print(f"  [Git] push 완료")
        else:
            print(f"  [Git] push 실패: {(push.stderr or '').strip()}")
    except Exception as e:
        print(f"  [Git] 오류: {e}")


# ============================================================
# 시장 스캔
# ============================================================
def _state_path(market):
    return os.path.join(STATE_DIR, f'{market}_result.json')


def scan_market(market, limit=0):
    t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f" {market.upper()} 스캔 시작 "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    instruments = local_instruments(market, limit)
    if not instruments:
        print('  [ERROR] 스캔할 CSV가 없습니다.')
        return None

    print(f"로컬 CSV 스캔 {len(instruments)}종목 ...")
    sig, watch = [], []
    asof = ''
    loaded = 0
    for n, (sym, info, path) in enumerate(instruments, 1):
        try:
            df = ST.load_csv(path)
            if df is None or len(df) < ST.MIN_BARS:
                continue
            loaded += 1
            r = evaluate(sym, info, df)
        except Exception:
            continue
        last = pd.Timestamp(df['Date'].values[-1]).strftime('%Y-%m-%d')
        asof = max(asof, last)
        if r is None:
            continue
        div_id = f'{market}_{r["tier"]}_{sym}'
        r['_card'] = render_card(r, div_id)
        del r['df']
        del r['idx']
        (sig if r['tier'] == 'signal' else watch).append(r)
        if n % 250 == 0:
            print(f'  {n}/{len(instruments)} 처리 · 후보 {len(sig) + len(watch)}개', flush=True)

    sig.sort(key=lambda r: -r['fib_pos'])
    watch.sort(key=lambda r: r['cci'])

    result = {
        'market': market,
        'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'scanned': loaded,
        'asof': asof,
        'signals': sig,
        'watch': watch,
    }

    os.makedirs(STATE_DIR, exist_ok=True)
    with open(_state_path(market), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"  진입 신호 {len(sig)}개 / 관찰 {len(watch)}개 "
          f"({time.time() - t0:.0f}s)")
    return result


def load_all_results():
    results = {}
    for market in ('kr', 'us'):
        path = _state_path(market)
        if os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as f:
                    results[market] = json.load(f)
            except Exception:
                results[market] = None
        else:
            results[market] = None
    return results


def build_archive_html():
    """public/ 내 날짜별 HTML 파일 목록을 아카이브 페이지로 생성."""
    import re
    date_re = re.compile(r'^(\d{4}-\d{2}-\d{2})\.html$')
    entries = []
    for fn in os.listdir(OUTPUT_DIR):
        m = date_re.match(fn)
        if m:
            entries.append(m.group(1))
    entries.sort(reverse=True)

    rows = '\n'.join(
        f'<tr><td><a href="{d}.html">{d}</a></td></tr>' for d in entries
    ) if entries else '<tr><td>아직 기록이 없습니다.</td></tr>'

    return (
        '<!DOCTYPE html>\n<html lang="ko"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>CCI 반등 모니터 — 아카이브</title>'
        f'<style>{CSS}</style></head><body>'
        '<header><h1>CCI 반등 모니터 — 아카이브</h1>'
        '<nav><a href="index.html">최신</a></nav></header>'
        '<div style="margin:20px"><table class="tbl">'
        '<thead><tr><th>날짜</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
        '</body></html>'
    )


def rebuild_html(results=None):
    if results is None:
        results = load_all_results()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    watchlist = generate_watchlist(results)
    html = build_html(results, watchlist)

    today = datetime.now().strftime('%Y-%m-%d')
    dated_path = os.path.join(OUTPUT_DIR, f'{today}.html')
    with open(dated_path, 'w', encoding='utf-8') as f:
        f.write(html)

    index_path = os.path.join(OUTPUT_DIR, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html)
    size_mb = os.path.getsize(index_path) / 1024 / 1024
    print(f"  HTML 저장: {dated_path} ({size_mb:.1f} MB)")

    archive_html = build_archive_html()
    archive_path = os.path.join(OUTPUT_DIR, 'archive.html')
    with open(archive_path, 'w', encoding='utf-8') as f:
        f.write(archive_html)
    n_dates = len([e for e in os.listdir(OUTPUT_DIR)
                   if e[:4].isdigit() and e.endswith('.html')])
    print(f"  아카이브: {archive_path} ({n_dates}일)")

    wl_path = os.path.join(OUTPUT_DIR, 'watchlist.txt')
    with open(wl_path, 'w', encoding='utf-8') as f:
        f.write(watchlist)
    n_wl = len(watchlist.splitlines()) if watchlist.strip() else 0
    print(f"  워치리스트: {wl_path} ({n_wl}종목)")


# ============================================================
# 스케줄러
# ============================================================
_ran_today = {}


def _already_ran(m):
    return _ran_today.get(m) == datetime.now().strftime('%Y%m%d')


def _mark_ran(m):
    _ran_today[m] = datetime.now().strftime('%Y%m%d')


def _is_weekend_sleep(now):
    wd = now.weekday()
    if wd == 6:
        wake = now + timedelta(days=1)
        wake = wake.replace(hour=KR_SCAN_TIME[0], minute=KR_SCAN_TIME[1],
                            second=0, microsecond=0)
        return True, wake
    if wd == 5:
        cutoff = now.replace(hour=US_SCAN_TIME[0], minute=US_SCAN_TIME[1],
                             second=0, microsecond=0)
        if now >= cutoff and _already_ran('us'):
            wake = now + timedelta(days=2)
            wake = wake.replace(hour=KR_SCAN_TIME[0],
                                minute=KR_SCAN_TIME[1],
                                second=0, microsecond=0)
            return True, wake
    return False, now


def _secs_until(h, m, only_days=None):
    now = datetime.now()
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    if only_days:
        while target.weekday() not in only_days:
            target += timedelta(days=1)
    return (target - now).total_seconds()


def update_chart_csv(market):
    """장 마감 뒤 원본 CSV 갱신을 성공한 경우에만 스캔한다."""
    script = BACKTEST_DIR / 'update_charts.py'
    if not script.exists():
        raise FileNotFoundError(f'업데이트 스크립트 없음: {script}')
    print(f'  [{market.upper()}] update_charts.py 실행')
    result = subprocess.run(
        [sys.executable, str(script), '--market', market, '--skip-monitor'],
        cwd=str(BACKTEST_DIR), text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f'CSV 업데이트 실패 (exit {result.returncode})')


def update_then_scan(market, git_message):
    update_chart_csv(market)
    if scan_market(market) is None:
        raise RuntimeError('스캔 결과가 없습니다.')
    rebuild_html()
    git_push(git_message)


def daemon_loop():
    print(f"{'=' * 60}")
    print(f" CCI 반등 모니터 24시간 스케줄러")
    print(f" 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")
    print(f"  한국:  평일       "
          f"{KR_SCAN_TIME[0]:02d}:{KR_SCAN_TIME[1]:02d} KST")
    print(f"  미국:  화~토      "
          f"{US_SCAN_TIME[0]:02d}:{US_SCAN_TIME[1]:02d} KST")
    print(f"  Git push: {ENABLE_GIT_PUSH}")
    print(f"  Ctrl+C 로 종료\n")

    print(f"  [시작] KR/US CSV 갱신 후 초기 스캔 실행")
    for m in ('kr', 'us'):
        try:
            update_then_scan(m, f'[scan] 초기 {m.upper()} {datetime.now():%Y%m%d_%H%M}')
            _mark_ran(m)
        except Exception as e:
            print(f"[ERROR] {m.upper()} 초기 스캔: {e}")

    SLEEP = 30

    while True:
        now = datetime.now()
        sl, wake = _is_weekend_sleep(now)
        if sl:
            secs = (wake - now).total_seconds()
            print(f"\r  [{now.strftime('%H:%M:%S')}] "
                  f"주말 휴식 ... 재개: {wake.strftime('%m/%d %H:%M')} "
                  f"({secs / 3600:.1f}h)          ",
                  end='', flush=True)
            time.sleep(SLEEP)
            continue

        h, m = now.hour, now.minute
        scanned = False

        if ((h, m) == KR_SCAN_TIME
                and now.weekday() < 5 and not _already_ran('kr')):
            _mark_ran('kr')
            try:
                update_then_scan('kr', f'[scan] KR {now.strftime("%Y%m%d")}')
                scanned = True
            except Exception as e:
                print(f"[ERROR] KR scan: {e}")

        elif ((h, m) == US_SCAN_TIME
              and now.weekday() in US_SCAN_DAYS
              and not _already_ran('us')):
            _mark_ran('us')
            try:
                update_then_scan('us', f'[scan] US {now.strftime("%Y%m%d")}')
                scanned = True
            except Exception as e:
                print(f"[ERROR] US scan: {e}")

        if not scanned:
            ks = _secs_until(*KR_SCAN_TIME, only_days={0, 1, 2, 3, 4})
            us = _secs_until(*US_SCAN_TIME, only_days=US_SCAN_DAYS)
            labels = [(ks, 'KR'), (us, 'US')]
            labels.sort(key=lambda x: x[0])
            nlbl = labels[0][1]
            ns = labels[0][0]
            print(f"\r  [{now.strftime('%H:%M:%S')}] 대기 중 ... "
                  f"다음: {nlbl} {ns / 3600:.1f}h 후          ",
                  end='', flush=True)

        time.sleep(SLEEP)


# ============================================================
# 메인
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description='CCI 반등 모니터링 + Vercel 배포')
    ap.add_argument('--market', choices=['kr', 'us', 'all'], default='all')
    ap.add_argument('--daemon', action='store_true',
                    help='24시간 스케줄러 모드')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--after-update', choices=['kr', 'us'],
                    help='update_charts.py 완료 뒤 해당 시장만 스캔')
    ap.add_argument('--no-git', action='store_true',
                    help='Git push 안 함')
    ap.add_argument('--open', action='store_true',
                    help='생성 후 브라우저로 열기')
    ap.add_argument('--strict', action='store_true',
                    help='백테스트 동일 조건 (추가 필터 전부 켜기)')
    args = ap.parse_args()

    global ENABLE_GIT_PUSH
    if args.no_git:
        ENABLE_GIT_PUSH = False

    if args.strict:
        ST.RECOVER_WINDOW     = _ORIG_RECOVER_WINDOW
        ST.USE_ROOM_FILTER    = True
        ST.USE_ATR_FILTER     = True
        ST.USE_DRYUP_FILTER   = True
        ST.USE_FIBNEAR_FILTER = True
        ST.USE_CLOSE_GT_MA5   = True
        print("  [STRICT] 백테스트 동일 조건으로 스캔")

    if args.daemon:
        daemon_loop()
        return

    markets = ([args.after_update] if args.after_update
               else (['kr', 'us'] if args.market == 'all' else [args.market]))
    for m in markets:
        scan_market(m, limit=args.limit)

    rebuild_html()

    now_str = datetime.now().strftime('%Y%m%d_%H%M')
    git_push(f'[scan] {",".join(m.upper() for m in markets)} {now_str}')

    if args.open:
        import webbrowser
        webbrowser.open(
            Path(os.path.join(OUTPUT_DIR, 'index.html')).as_uri())


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n[종료] Ctrl+C")
