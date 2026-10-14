from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

BASE = Path.cwd() / 'metro_logistics_qvns_nsga2_package' / 'metro_logistics_qvns_nsga2_package'
OUT = BASE / 'results' / '5.20' / 'ch5_visual_updates'
PROC = OUT / 'process_data'
OUT.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

PALETTE = {
    'baseline': '#7A7F87',
    'cost': '#3B73B9',
    'balanced': '#D98C2B',
    'service': '#4C9A6A',
    'grid': '#E6E8EF',
    'axis': '#4A4F59',
    'spine': '#D1D5DE',
}


def pick_font() -> str:
    preferred = [
        'Microsoft YaHei', 'SimHei', 'Source Han Sans SC', 'Noto Sans CJK SC',
        'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans'
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in preferred:
        if font in available:
            return font
    return 'DejaVu Sans'

FONT = pick_font()
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [FONT, 'DejaVu Sans', 'Arial'],
    'axes.unicode_minus': False,
    'figure.dpi': 160,
    'savefig.dpi': 450,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': PALETTE['spine'],
    'axes.labelcolor': PALETTE['axis'],
    'xtick.color': PALETTE['axis'],
    'ytick.color': PALETTE['axis'],
    'text.color': '#222222',
    'legend.frameon': False,
    'axes.titleweight': 'semibold',
    'axes.titlesize': 10.5,
    'axes.labelsize': 9.5,
    'xtick.labelsize': 8.4,
    'ytick.labelsize': 8.4,
    'legend.fontsize': 8.5,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})


def style_axes(ax, axis='y'):
    ax.grid(True, axis=axis, color=PALETTE['grid'], linewidth=0.8, alpha=0.95)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PALETTE['spine'])
    ax.spines['bottom'].set_color(PALETTE['spine'])
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)


def add_bar_labels(ax, bars, values, fmt, pad_ratio=0.018):
    ymax = max(values) if max(values) > 0 else 1
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * pad_ratio,
                fmt.format(val), ha='center', va='bottom', fontsize=8.0)


def save_fig(fig, stem):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{stem}.{ext}', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def build_preference_baseline_data() -> pd.DataFrame:
    df = pd.DataFrame([
        {'方案': '全公路直送', '解编号': '-', '综合成本_元日': 204130.69, '综合成本_万元日': 20.41,
         '地铁服务比例': 0.0, '公路直送量': 66970.02, '碳排放_tCO2日': 0.510199, '相对全公路综合成本变化': 0.0,
         '类型': 'baseline'},
        {'方案': '成本偏好方案', '解编号': '57', '综合成本_元日': 202113.69, '综合成本_万元日': 20.21,
         '地铁服务比例': 0.3627, '公路直送量': 42681.11, '碳排放_tCO2日': 0.340199, '相对全公路综合成本变化': -0.009881,
         '类型': 'cost'},
        {'方案': '均衡偏好方案', '解编号': '68', '综合成本_元日': 207005.43, '综合成本_万元日': 20.70,
         '地铁服务比例': 0.4239, '公路直送量': 38582.15, '碳排放_tCO2日': 0.327277, '相对全公路综合成本变化': 0.01408286,
         '类型': 'balanced'},
        {'方案': '服务偏好方案', '解编号': '85', '综合成本_元日': 215978.24, '综合成本_万元日': 21.60,
         '地铁服务比例': 0.5117, '公路直送量': 32699.76, '碳排放_tCO2日': 0.307767, '相对全公路综合成本变化': 0.058039,
         '类型': 'service'},
    ])
    baseline_direct = float(df.loc[df['方案'] == '全公路直送', '公路直送量'].iloc[0])
    baseline_carbon = float(df.loc[df['方案'] == '全公路直送', '碳排放_tCO2日'].iloc[0])
    df['公路直送量降幅'] = 1 - df['公路直送量'] / baseline_direct
    df['碳减排率'] = 1 - df['碳排放_tCO2日'] / baseline_carbon
    df.to_csv(PROC / 'fig5_10_preference_baseline_comparison.csv', index=False, encoding='utf-8-sig')
    return df


def fig_5_10(df: pd.DataFrame) -> None:
    labels = [f"{r['方案']}\n解{r['解编号']}" if r['解编号'] != '-' else r['方案'] for _, r in df.iterrows()]
    x = np.arange(len(df))
    colors = [PALETTE[t] for t in df['类型']]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), constrained_layout=True)
    fig.suptitle('图 5-10 不同决策偏好下地铁联运方案与全公路基准方案关键指标对比图',
                 fontsize=13, fontweight='semibold', y=1.025)

    ax = axes[0, 0]
    vals = df['综合成本_万元日'].to_numpy()
    bars = ax.bar(x, vals, color=colors, width=0.58, edgecolor='white', linewidth=0.8)
    ax.set_title('(a) 综合运营成本与相对成本变化')
    ax.set_ylabel('综合运营成本（万元/日）')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(19.7, 22.1)
    style_axes(ax)
    add_bar_labels(ax, bars, vals, '{:.2f}', pad_ratio=0.01)
    ax2 = ax.twinx()
    cost_change = df['相对全公路综合成本变化'].to_numpy() * 100
    ax2.plot(x, cost_change, color='#B33E3E', marker='o', linewidth=1.8, markersize=5.2,
             label='相对全公路综合成本变化')
    ax2.axhline(0, color='#B33E3E', linewidth=0.8, linestyle='--', alpha=0.65)
    ax2.set_ylabel('相对全公路综合成本变化（%）', color='#B33E3E')
    ax2.tick_params(axis='y', colors='#B33E3E')
    ax2.set_ylim(-2.2, 7.0)
    for xi, v in zip(x, cost_change):
        ax2.text(xi, v + (0.35 if v >= 0 else -0.55), f'{v:+.2f}%', ha='center',
                 va='bottom' if v >= 0 else 'top', fontsize=7.7, color='#B33E3E')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color('#D8B3B3')

    ax = axes[0, 1]
    vals = df['地铁服务比例'].to_numpy() * 100
    bars = ax.bar(x, vals, color=colors, width=0.58, edgecolor='white', linewidth=0.8)
    ax.set_title('(b) 地铁联运服务比例')
    ax.set_ylabel('地铁联运服务比例（%）')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 60)
    style_axes(ax)
    add_bar_labels(ax, bars, vals, '{:.2f}%')

    ax = axes[1, 0]
    vals = df['公路直送量'].to_numpy() / 10000
    bars = ax.bar(x, vals, color=colors, width=0.58, edgecolor='white', linewidth=0.8)
    ax.set_title('(c) 公路直送量及道路交通减负')
    ax.set_ylabel('公路直送量（万件/日）')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 7.4)
    style_axes(ax)
    add_bar_labels(ax, bars, vals, '{:.2f}')
    for i, r in df.iterrows():
        if r['方案'] != '全公路直送':
            ax.text(i, vals[i] + 0.38, f"降幅 {r['公路直送量降幅']*100:.2f}%", ha='center', va='bottom', fontsize=7.5, color='#3F4B5B')

    ax = axes[1, 1]
    vals = df['碳排放_tCO2日'].to_numpy()
    bars = ax.bar(x, vals, color=colors, width=0.58, edgecolor='white', linewidth=0.8)
    ax.set_title('(d) 碳排放与碳减排绩效')
    ax.set_ylabel('碳排放（tCO2/日）')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.58)
    style_axes(ax)
    add_bar_labels(ax, bars, vals, '{:.3f}')
    for i, r in df.iterrows():
        if r['方案'] != '全公路直送':
            ax.text(i, vals[i] + 0.035, f"减排 {r['碳减排率']*100:.2f}%", ha='center', va='bottom', fontsize=7.5, color='#3F4B5B')

    handles = [plt.Line2D([0], [0], marker='s', linestyle='', markerfacecolor=PALETTE[k],
                          markeredgecolor='white', markersize=8, label=v)
               for k, v in [('baseline', '全公路直送基准方案'), ('cost', '成本偏好方案'),
                            ('balanced', '均衡偏好方案'), ('service', '服务偏好方案')]]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.035))
    save_fig(fig, 'fig_5_10_preference_baseline_key_metrics')


def fig_5_station_load_balanced() -> None:
    src = BASE / 'results' / 'chapter5_q11_results' / 'representative_solution' / 'qvns_nsga2_station_load.csv'
    df = pd.read_csv(src)
    df = df[df['是否启用'] == 1].sort_values('负载率', ascending=True)
    df.to_csv(PROC / 'fig5_balanced_preference_station_load.csv', index=False, encoding='utf-8-sig')
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = ['#4C9A6A' if v < 0.83 else '#D98C2B' if v < 0.85 else '#B33E3E' for v in df['负载率']]
    ax.barh(df['地铁站名称'], df['负载率'] * 100, color=colors, edgecolor='white', height=0.56)
    ax.axvline(85, color='#B33E3E', linestyle='--', linewidth=1.2, label='最高可用负载率 85%')
    ax.set_title('均衡偏好方案下地铁物流配送转运点负载率')
    ax.set_xlabel('负载率（%）')
    ax.set_ylabel('启用地铁物流配送转运点')
    ax.set_xlim(0, 92)
    style_axes(ax, axis='x')
    for y, (_, row) in enumerate(df.iterrows()):
        ax.text(row['负载率'] * 100 + 0.5, y, f"{row['分配量_件日']:.0f}/{row['容量_件日']:.0f}", va='center', fontsize=8.5)
    ax.legend(loc='lower right')
    save_fig(fig, 'fig_5_balanced_preference_station_load')


def fig_6_line_capacity_balanced() -> None:
    src = BASE / 'results' / 'chapter5_q11_results' / 'representative_solution' / 'qvns_nsga2_line_load.csv'
    df = pd.read_csv(src)
    pivot = df.pivot_table(index='线路', columns='班次', values='班次负载率', aggfunc='mean').fillna(0)
    pivot = pivot.loc[sorted(pivot.index, key=lambda x: int(str(x).replace('号线', '')))]
    pivot.to_csv(PROC / 'fig6_balanced_preference_line_trip_capacity.csv', encoding='utf-8-sig')
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('load', ['#F7F8FA', '#79B8D9', '#F1C45B', '#C84B4B'])
    im = ax.imshow(pivot.values * 100, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    ax.set_title('均衡偏好方案下客运平峰期列车班次剩余运力占用情况')
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f'班次{c}' for c in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j] * 100
            if val > 0:
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8,
                        color='white' if val > 65 else '#222222')
    ax.set_xlabel('可用客运平峰期列车班次')
    ax.set_ylabel('地铁线路')
    for side in ['left', 'right', 'top', 'bottom']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(PALETTE['spine'])
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label('列车班次剩余运力占用率（%）')
    save_fig(fig, 'fig_6_balanced_preference_line_trip_capacity_heatmap')


if __name__ == '__main__':
    data = build_preference_baseline_data()
    fig_5_10(data)
    fig_5_station_load_balanced()
    fig_6_line_capacity_balanced()
    print('Generated files in:', OUT)
    print('Font:', FONT)
