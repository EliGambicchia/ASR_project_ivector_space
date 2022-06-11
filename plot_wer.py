import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle

sns.set_style("whitegrid")
colours = sns.color_palette("tab10")
palette = {"US": colours[0], "CAN": colours[8], "ENG": colours[3], "IND": colours[1], "AUS": colours[4],
           "AFR": colours[9], "NZL": colours[6], "PHI": colours[7], "IRE": colours[2], "SCO": colours[5]}

# BASELINE WITHOUT AND WITH IVECTORS ON TEST SET BASELINE
df_test_baseline = pd.DataFrame({
    'US': [39.6, 33.3],
    'ENG': [40.2, 37.9],
    'IND': [57.6, 56.8],
    'CAN': [37.7, 34.4],
    'AUS': [40.2, 38],
    'AFR': [40.6, 37.6],
    'NZL': [40.8, 34.0],
    'IRE': [39.7, 37.5],
    'SCO': [46.6, 41.2],
    'PHI': [48.4, 48],
}, index=['TDNN\nwithout i-vectors', 'TDNN\nwith i-vectors'])

# DATA WITH DEV AND TEST (i-vectors with baseline, diversity, utterances)
df_devtest_bdu = pd.DataFrame({
   'IRE': [42.2, 37.5, 37.8, 37.0, 36.8, 31.0],
   'ENG': [30.3, 37.9, 38.8, 39.8, 38.5, 38.7],
    'CAN': [32.9, 34.4, 33.2, 34.2, 28.6, 27.6],
    'AFR': [33.5, 37.6, 32.4, 33.5, 24.1, 36.9],
    'US': [29.8, 33.3, 31.8, 34.2, 38.3, 40.4],
    'PHI': [47.6, 48.0, 45.3, 41.5, 42.4, 37.8],
    'AUS': [38.2, 38.0, 31.1, 31.2, 27.1, 28.8],
    'SCO': [36.7, 41.2, 32.8, 38.8, 28.2, 34.6],
    'IND': [42.7, 56.8, 50.7, 51.1, 42.2, 50.6],
    'NZL': [42.4, 34.0, 39.3, 33.4, 35.6, 30.9],
}, index=['dev_b', 'test_b', 'dev_d', 'test_d', 'dev_u', 'test_u'])

# WITH TEST ONLY (from baseline, diversity, utterances)
df_testonly_bdu = pd.DataFrame({
    'IRE': [37.5, 37.0, 31.0],
    'ENG': [37.9, 39.8, 38.7],
    'CAN': [34.4, 34.2, 27.6],
    'AFR': [37.6, 33.5, 36.9],
    'US': [33.3, 34.2, 40.4],
    'PHI': [48.0, 41.5, 37.8],
    'AUS': [38.0, 31.2, 28.8],
    'SCO': [41.2, 38.8, 34.6],
    'IND': [56.8, 51.1, 50.6],
    'NZL': [34.0, 33.4, 30.9],
}, index=['Baseline', 'Diversity', 'Jumbo'])

df_jumbo_with_speed = pd.DataFrame({
    'USA': [33.3, 34.2, 40.4, 40.3],
    'England': [37.9, 39.8, 38.7, 38],
    'India': [56.8, 51.1, 50.6, 49.8],
    'Canada': [34.4, 34.2, 27.6, 25.7],
    'Australia': [38.0, 31.2, 28.8, 28.7],
    'Africa': [37.6, 33.5, 36.9, 36.9],
    'New Zealand': [34.0, 33.4, 30.9, 31.2],
    'Ireland': [37.5, 37.0, 31.0, 31.1],
    'Scotland': [41.2, 38.8, 34.6, 33.3],
    'Philippines': [48.0, 41.5, 37.8, 34.2],
}, index=['Baseline', 'Diversity', 'Jumbo', 'Jumbo +\nspeed aug.'])

# DATA FOR DEV AND TEST SETS
def plot_lines(df, label, title, y_limit):

    g = sns.lineplot(data=df, markers=True, dashes=False, palette=palette, markersize=12)
    plt.legend(title='English variants', bbox_to_anchor=(1.01, 1),
               borderaxespad=0)

    plt.title(title)
    g.set(ylim=(y_limit, 60))

    plt.xlabel("Models", size=15)
    plt.ylabel("WER (%)", size=15)

    plt.tight_layout()
    plt.savefig(f"/Users/macbookpro/Desktop/University of Edinburgh/Dissertation/multitask_folders/visualisation/{label}.pdf")


def plot_wer_gmm(label):
    # DATA FOR GMM-HMM SYSTEMS
    data_b = {'type_gmm': ['Triphones', 'Triphones \u0394 + \u0394\u0394', 'LDA+MLLT', 'LDA+MLLT+SAT'],
              'type_filter': ['Baseline', 'Baseline', 'Baseline', 'Baseline'],
              'wer': [57.30, 56.71, 53.07, 46.54]}

    data_d = {'type_gmm': ['Triphones', 'Triphones \u0394 + \u0394\u0394', 'LDA+MLLT', 'LDA+MLLT+SAT'],
              'type_filter': ['Diversity', 'Diversity', 'Diversity', 'Diversity'],
              'wer': [57.67, 56.95, 53.94, 48.05]}

    data_u = {'type_gmm': ['Triphones', 'Triphones \u0394 + \u0394\u0394', 'LDA+MLLT', 'LDA+MLLT+SAT'],
              'type_filter': ['Jumbo', 'Jumbo', 'Jumbo', 'Jumbo'],
              'wer': [56.93, 55.86, 51.55, 46.12]}

    df_b = pd.DataFrame.from_dict(data_b)
    df_d = pd.DataFrame.from_dict(data_d)
    df_u = pd.DataFrame.from_dict(data_u)
    df_gmm = pd.concat([df_b, df_d, df_u], axis=0, ignore_index=True)
    print(df_gmm)

    plt.figure(figsize=(18, 5))

    sns.set(font_scale=1.4)

    g = sns.FacetGrid(df_gmm, col="type_gmm", height=4, aspect=.9, col_wrap=4, margin_titles=True, sharex=False)

    g.map(sns.barplot, "type_filter", "wer", order=["Baseline", "Diversity", "Jumbo"], palette="colorblind")

    g.set(ylim=(40, 60))

    g.set_xlabels("Datasets", size=20)
    g.set_ylabels("WER (%)", size=20)

    # g.set_xticklabels(['Baseline', 'Diversity', 'Jumbo'], size=18)
    # g.set_yticklabels(g.get_yticks(), size=18)


    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Results of the GMM-HMM systems", size=25)

    g.set_titles('{col_name}', size=25)

    plt.tight_layout()
    plt.savefig(f"/Users/macbookpro/Desktop/University of Edinburgh/Dissertation/multitask_folders/visualisation/{label}.pdf")
    #plt.show()

def show_hatches(figure):
    #hatches = ['-', '\\', '+', 'x', '*', 'o']
    hatches = cycle(['-', '//', 'x'])
    # Loop over the bars
    for i, patch in enumerate(figure.patches):
        # Blue bars first, then green bars
        hatch = next(hatches)
        patch.set_hatch(hatch)
    #
    # num_locations = len(tips.day.unique())

    # for i, bar in enumerate(figure.patches):
    #     # Set a different hatch for each bar
    #     hatch = next(hatches)
    #     bar.set_hatch(hatches[i])

    figure.legend(loc='best')

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            # _x = p.get_x() + p.get_width() / 2
            # _y = p.get_y() + p.get_height()
            # value = '{:.1f}'.format(p.get_height())
            # ax.text(_x, _y, value, ha="center", va="top", size=35, rotation=90, color="white")
            # #
            _x = p.get_x() + p.get_width()
            _y = p.get_y() + p.get_height() / 2
            value = '{:.1f}'.format(p.get_width())
            ax.text(_x, _y, value, ha="right", va="center_baseline", size=28, color="white", weight='bold')


    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

# DATAFRAMES
df_two_accents = pd.DataFrame({
    'accent': ['Avg', 'US', 'ENG', 'IND', 'CAN', 'AUS', 'AFR', 'NZL', 'IRE', 'SCO', 'PHI'],
    'All data': [35.7, 40.4, 38.7, 50.6, 27.6, 28.8, 36.9, 30.9, 31.0, 34.6, 37.8],
    'W/out CAN': [35.3, 36.7, 37.1, 51.3, 27.3, 28.6, 36.4, 31.8, 31.2, 34.7, 36.5],
    'W/out IND': [35.8, 38.4, 38.2, 57.5, 25, 28.7, 35.4, 31.1, 29.9, 34.2, 38.2]
})

df_speed = pd.DataFrame({
    'accent': ['Avg', 'US', 'ENG', 'IND', 'CAN', 'AUS', 'AFR', 'NZL', 'IRE', 'SCO', 'PHI'],
    'Without speed augmentation': [35.7, 40.4, 38.7, 50.6, 27.6, 28.8, 36.9, 30.9, 31.0, 34.6, 37.8],
    'With speed augmentation': [35.0, 40.3, 38, 49.8, 25.7, 28.7, 36.9, 31.2, 31.1, 33.3, 34.2]
})

# CANADA VS INDIA
def plot_wer_two_accents(df, label, title_legend):

    fig, ax1 = plt.subplots(figsize=(17, 17))
    sns.set_style("whitegrid")

    tidy = df.melt(id_vars='accent').rename(columns=str.title)

    sns.barplot(x='Value', y='Accent', hue='Variable', data=tidy, ax=ax1, palette='colorblind')
    sns.despine(fig)

    #ax1.set(ylim=(20, 60))
    plt.xlabel("WER (%)", size=40)
    plt.ylabel("Accents", size=40)
    ax1.set_yticklabels(['Avg', 'US', 'ENG', 'IND', 'CAN', 'AUS', 'AFR', 'NZL', 'IRE', 'SCO', 'PHI'], size=35)
    ax1.set_xticklabels(ax1.get_xticks(), size=35)

    plt.legend(loc="best", title=title_legend, title_fontsize=40, frameon=True, fontsize=30, bbox_to_anchor=(1.05, 1))
    show_values_on_bars(ax1)

    #show_hatches(ax1)

    plt.tight_layout()
    #plt.show()
    plt.savefig(f"/Users/macbookpro/Desktop/University of Edinburgh/Dissertation/multitask_folders/visualisation/{label}.pdf")


#plot_wer_gmm(label='gmm-hmm-bdu')

#plot_wer_two_accents(df=df_two_accents, label="very_new_canada_vs_india", title_legend="Accent filtered")

#plot_wer_two_accents(df=df_speed, label='speed-bars', title_legend="Training of the i-vector extractor")


plot_lines(df_test_baseline, title="", label= "WER_test_baseline_with_without_ivectors", y_limit=30)
#plot_lines(df_testonly_bdu, title="", label= "WER_3way", y_limit=25)
