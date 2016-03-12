import logging
import argparse
import os

from flask import Flask, render_template
from flask.ext.bootstrap import Bootstrap

from bokeh.plotting import figure, ColumnDataSource
from bokeh.embed import components
from bokeh.charts import Bar
from bokeh.models import (HoverTool, BoxZoomTool, ResetTool, TapTool, CustomJS,
                          BoxSelectTool)

from sklearn.preprocessing import Imputer, scale
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import gzip
import operator

app = Flask(__name__)
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
log = app.logger
Bootstrap(app)

# connection_string = os.getenv("DATABASE_URL")
# engine = create_engine(connection_string)

def kpi_ranks(df, kommun, missing):
    rank = {}
    for kpi in df.columns:
        if(missing[kpi][kommun]==True): continue
        all_m = sorted(df[kpi])
        this_m = df[kpi][kommun]
        rank[kpi] = all_m.index(this_m)
    return(rank)

def load_data():
    original_data = pd.read_csv(gzip.open("data/data.csv.gz"), sep="\t")
    return(original_data)
    

def compute_principal_components(d):
    log.debug("Computing principal components")
    
    mx = d.pivot(index='municipality_name', columns='kpi', values='value')
    # Imputation. First replace 'None' string with NaN
    mis = mx == 'None'
    mx[mis] = np.nan
    missing = pd.isnull(mx)

    imp = Imputer(strategy="mean", axis=1)
    imp.fit(mx)
    # Impute mean values
    imputed_df = pd.DataFrame(imp.transform(mx), index=mx.index, columns=mx.columns.values)
    colvar = imputed_df.var(axis=0)
    # Remove constant columns
    nonconstant_df = imputed_df[colvar[colvar > 0.01].index]
    # Scale values
    scaled_df = pd.DataFrame(scale(nonconstant_df), index=mx.index, columns=nonconstant_df.columns.values)

    pca = PCA(n_components=2)
    pc = pca.fit(scaled_df).transform(scaled_df)
    log.debug("Finished computing principal compoments")
    return scaled_df, pc, missing


def create_pca_plot():
    """Render the initial PCA plot with all regions."""
    callback = CustomJS(code="""
        var indexes = cb_obj.get('selected')['1d'].indices;
        var descs = [];
        for (index in indexes) {
            descs.push(cb_obj.get('data').desc[indexes[index]]);
        }
        window.vue.$data.regions = descs;
    """)

    tools = [HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("desc", "@desc"),
        ]
    ), BoxZoomTool(), ResetTool(), TapTool(callback=callback), BoxSelectTool(callback=callback)]

    source = ColumnDataSource(
        data=dict(
            x=pc[:, 0],
            y=pc[:, 1],
            desc=df3.index,
        )
    )

    plot = figure(title='PCA plot', plot_width=960, plot_height=600,
              tools=tools, responsive=True)

    plot.background_fill_color = 'beige'
    plot.background_fill_alpha = 0.5

    renderer = plot.circle('x', 'y', size=10, source=source,
                           # set visual properties for selected glyphs
                           selection_color="firebrick",

                           # set visual properties for non-selected glyphs
                           nonselection_fill_alpha=0.2,
                           nonselection_fill_color="blue",
                           nonselection_line_color="firebrick",
                           nonselection_line_alpha=1.0)

    plot.xaxis.axis_label = "x"
    plot.yaxis.axis_label = "y"

    pca_fig_js, pca_fig_div = components(plot)
    return pca_fig_js, pca_fig_div

def create_bar_plot():

    municipality = "Uppsala"
    #orig_copy = original_data.copy()
    #orig_copy["municipality_name"] = orig_copy.index
    #print(orig_copy.index)

    rks = kpi_ranks(df3, municipality, missing)
    sorted_ranks = sorted(rks.items(), key=operator.itemgetter(1), reverse=True)
    top_kpi = [k[0] for k in sorted_ranks[:10]]

    #small_data_melted = pd.melt(orig_copy, id_vars='municipality_name')
    selected = original_data[(original_data.kpi.isin(top_kpi)) & (original_data.municipality_name == municipality)]
    selected.value = selected.value.astype(float)

    print(selected)

    bar_plot = Bar(
        selected,
        'municipality_name',
        values='value',
        group='kpi',
        title="Test bar plot")

    bar_fig_js, bar_fig_div = components(bar_plot)
    return bar_fig_js, bar_fig_div

original_data = load_data()
df3, pc, missing = compute_principal_components(original_data)

@app.route('/')
def index():
    pca_fig_js, pca_fig_div = create_pca_plot()
    bar_fig_js, bar_fig_div = create_bar_plot()

    return \
        render_template(
            "figures.html",
            pcaFigJs=pca_fig_js,
            pcaFigDiv=pca_fig_div,
            barFigJs=bar_fig_js,
            barFigDiv=bar_fig_div)

def start():
    # Get port form environment or default to 5000
    port = int(os.getenv("PORT", 5000))
    # Check for env. variable or start on localhost
    interface = os.getenv("INTERFACE", "localhost")
    app.run(host=interface, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        app.debug = True
        app.logger.setLevel(logging.DEBUG)

    start()

