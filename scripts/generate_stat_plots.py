import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports', 'stats')
os.makedirs(BASE, exist_ok=True)

def save_dispersion_plot(df, out):
    # single-row summary: show dispersion ratio as a big number
    val = df.loc[0, 'dispersion_ratio']
    fig, ax = plt.subplots(figsize=(4,2))
    ax.axis('off')
    ax.text(0.5, 0.5, f"Dispersion ratio\n{val:.2f}", ha='center', va='center', fontsize=18)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def save_promotions_plot(df, out):
    # filter promotions and pick top 10 by uplift_pct
    dfp = df[df['event_type']=='promotion'].copy()
    if dfp.empty:
        return
    dfp = dfp.sort_values('uplift_pct', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(dfp['cluster'].astype(str), dfp['uplift_pct'])
    ax.set_xlabel('cluster')
    ax.set_ylabel('uplift_pct')
    ax.set_title('Top 10 promotion uplift_pct by cluster')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def save_residuals_plot(df, out):
    # plot residual_std by cluster
    dfc = df.copy()
    dfc['cluster'] = dfc['cluster'].astype(int)
    dfc = dfc.sort_values('cluster')
    fig, ax = plt.subplots(figsize=(9,3))
    ax.bar(dfc['cluster'].astype(str), dfc['residual_std'])
    ax.set_xlabel('cluster')
    ax.set_ylabel('residual_std')
    ax.set_title('Residual standard deviation by cluster')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def main():
    try:
        dist = pd.read_csv(os.path.join(BASE, 'distribution_count_diagnostics_overall.csv'))
    except Exception:
        dist = None
    try:
        effects = pd.read_csv(os.path.join(BASE, 'business_action_effects.csv'))
    except Exception:
        effects = None
    try:
        residuals = pd.read_csv(os.path.join(BASE, 'residual_diagnostics_by_cluster.csv'))
    except Exception:
        residuals = None

    if dist is not None:
        save_dispersion_plot(dist, os.path.join(BASE, 'dispersion_overall.png'))
    if effects is not None:
        save_promotions_plot(effects, os.path.join(BASE, 'promotions_top10_uplift.png'))
    if residuals is not None:
        save_residuals_plot(residuals, os.path.join(BASE, 'residuals_by_cluster.png'))

    print('Plots saved to', BASE)

if __name__ == '__main__':
    main()
