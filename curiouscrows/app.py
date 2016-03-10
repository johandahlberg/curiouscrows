

import logging
import argparse
import os
import sys

from flask import Flask, render_template

from bokeh.plotting import figure, ColumnDataSource
from bokeh.embed import components
from bokeh.models import HoverTool, BoxZoomTool, ResetTool, TapTool, CustomJS

from sklearn.preprocessing import Imputer, scale
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import gzip

from sqlalchemy import create_engine

app = Flask(__name__)
log = app.logger

# connection_string = os.getenv("DATABASE_URL")
# engine = create_engine(connection_string)


def compute_principal_components():
    log.debug("Computing principal components")
    # d = pd.read_sql_table("data", engine,
    #                       columns=["kpi", "municipality_name", "value"])
    d = pd.read_csv(gzip.open("data/data.csv.gz"),sep="\t")
    mx = d.pivot(index='municipality_name', columns='kpi', values='value')

    # Imputation. First replace 'None' string with NaN
    mis = mx == 'None'
    mx[mis] = np.nan

    imp = Imputer(strategy="mean", axis=1)
    imp.fit(mx)
    # Impute mean values
    df = pd.DataFrame(imp.transform(mx),index=mx.index)
    colvar = df.var(axis=0)
    # Remove constant columns
    df2 = df[colvar[colvar>0.01].index]
    # Scale values
    df3 = pd.DataFrame(scale(df2), index=mx.index)

    pca = PCA(n_components=2)
    pc = pca.fit(df3).transform(df3)
    log.debug("Finished computing principal compoments")
    return df3, pc


df3, pc = compute_principal_components()

@app.route('/')
def index():

    tools = [HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("desc", "@desc"),
        ]
    )]

    source = ColumnDataSource(
        data=dict(
            x=pc[:,0],
            y=pc[:,1],
            desc=df3.index,
        )
    )

    p = figure(title='PCA plot', plot_width=800, plot_height=500, tools=tools)

    p.circle('x', 'y', source=source, alpha=0.5)

    p.xaxis.axis_label = "x"
    p.yaxis.axis_label = "y"

    fig_js, fig_div = components(p)

    return(render_template(
        "figures.html",
        figJS=fig_js,
        figDiv=fig_div))


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

