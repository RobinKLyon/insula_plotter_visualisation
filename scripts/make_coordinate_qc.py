from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'data' / 'sites_user.csv'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
df = pd.read_csv(CSV)
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
pairs = [('x', 'y', 'Axial projection (X / Y)'), ('x', 'z', 'Coronal projection (X / Z)'), ('y', 'z', 'Sagittal projection (Y / Z)')]
lim = float(df['t_value'].abs().max())
for ax, (a, b, title) in zip(axes, pairs):
    sc = ax.scatter(df[a], df[b], c=df['t_value'], cmap='coolwarm', vmin=-lim, vmax=lim,
                    s=35, edgecolors='#222222', linewidths=.45)
    for _, row in df.iterrows():
        ax.annotate(row.site_id, (row[a], row[b]), xytext=(2, 2), textcoords='offset points', fontsize=5, alpha=.6)
    ax.set_title(title)
    ax.set_xlabel(f'{a.upper()} (mm)')
    ax.set_ylabel(f'{b.upper()} (mm)')
    ax.axvline(0, color='#888888', lw=.6, ls='--') if a == 'x' else None
    ax.grid(alpha=.18)
cb = fig.colorbar(sc, ax=axes, shrink=.78, pad=.02)
cb.set_label('t_value')
fig.suptitle('MNI152 coordinates — input QC only (not an anatomical surface projection)', fontsize=12)
fig.savefig(OUT / 'mni152_sites_coordinate_qc.png', dpi=220, facecolor='white')
plt.close(fig)

summary = pd.DataFrame([
    {'metric':'n_sites','value':len(df)},
    {'metric':'n_left_x_negative','value':int((df.x < 0).sum())},
    {'metric':'n_right_x_positive','value':int((df.x > 0).sum())},
    {'metric':'t_value_min','value':df.t_value.min()},
    {'metric':'t_value_max','value':df.t_value.max()},
    {'metric':'t_value_mean','value':df.t_value.mean()},
    {'metric':'t_value_median','value':df.t_value.median()},
])
summary.to_csv(OUT / 'input_data_summary.csv', index=False)
df['color'].value_counts().rename_axis('csv_color').reset_index(name='n_sites').to_csv(OUT / 'input_color_counts.csv', index=False)
print(summary.to_string(index=False))
print('\nColor counts:')
print(df['color'].value_counts().to_string())
