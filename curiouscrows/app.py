import logging
import argparse
import os

from flask import Flask, render_template
from flask.ext.bootstrap import Bootstrap

from bokeh.plotting import figure, ColumnDataSource
from bokeh.embed import components
from bokeh.models import (HoverTool, BoxZoomTool, ResetTool, TapTool, CustomJS)
from bokeh.charts import Bar

from sklearn.preprocessing import Imputer, scale
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import gzip

app = Flask(__name__)
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
log = app.logger
Bootstrap(app)

# connection_string = os.getenv("DATABASE_URL")
# engine = create_engine(connection_string)


def compute_principal_components():
    log.debug("Computing principal components")
    # d = pd.read_sql_table("data", engine,
    #                       columns=["kpi", "municipality_name", "value"])
    d = pd.read_csv(gzip.open("data/data.csv.gz"), sep="\t")
    mx = d.pivot(index='municipality_name', columns='kpi', values='value')

    # Imputation. First replace 'None' string with NaN
    mis = mx == 'None'
    mx[mis] = np.nan

    imp = Imputer(strategy="mean", axis=1)
    imp.fit(mx)
    # Impute mean values
    df = pd.DataFrame(imp.transform(mx), index=mx.index, columns=mx.columns.values)
    colvar = df.var(axis=0)
    # Remove constant columns
    df2 = df[colvar[colvar > 0.01].index]
    # Scale values
    df3 = pd.DataFrame(scale(df2), index=mx.index, columns=df2.columns.values)

    pca = PCA(n_components=2)
    pc = pca.fit(df3).transform(df3)
    log.debug("Finished computing principal compoments")
    return df3, pc


def create_pca_plot():
    """Render the initial PCA plot with all regions."""
    callback = CustomJS(code="""
        var index = cb_obj.get('selected')['1d'].indices[0];
        if (index !== undefined) {
            var desc = cb_obj.get('data').desc[index];
            history.pushState({}, '', '#' + desc);
        }
    """)

    tools = [HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("desc", "@desc"),
        ]
    ), BoxZoomTool(), ResetTool(), TapTool(callback=callback)]

    source = ColumnDataSource(
        data=dict(
            x=pc[:, 0],
            y=pc[:, 1],
            desc=df3.index,
        )
    )

    pca_figure = figure(title='PCA plot', plot_width=960, plot_height=600, tools=tools,
                        responsive=True)

    pca_figure.circle('x', 'y', size=10, fill_color='navy', source=source, alpha=0.5,
                      hover_fill_color="firebrick")

    pca_figure.xaxis.axis_label = "x"
    pca_figure.yaxis.axis_label = "y"

    pca_fig_js, pca_fig_div = components(pca_figure)
    return pca_fig_js, pca_fig_div

def create_bar_plot():

    small_data = df3.head()
    small_data["municipality_name"] = small_data.index

    small_data_melted = pd.melt(small_data, id_vars='municipality_name')

    small_data_melted = small_data_melted[
        (small_data_melted.variable == "N00002") |
        (small_data_melted.variable == "N00003") |
        (small_data_melted.variable == "N00005  ")
    ]

    bar_plot = Bar(
        small_data_melted,
        'municipality_name',
        values='value',
        group='variable',
        title="Test bar plot")

    bar_fig_js, bar_fig_div = components(bar_plot)
    return bar_fig_js, bar_fig_div


df3, pc = compute_principal_components()

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

