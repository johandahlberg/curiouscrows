import random
import argparse
import os

from flask import Flask, render_template

from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.embed import components
from bokeh.models import HoverTool

from sklearn.preprocessing import Imputer, scale
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import gzip

app = Flask(__name__)

@app.route('/')
def index():

    #y = [random.random() for n in range(50)]
    #x = [random.random() for n in range(50)]

    d = pd.read_csv(gzip.open("../data/data.csv.gz"),sep="\t")
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
    
    hover = HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("desc", "@desc"),
        ]
    )

    source = ColumnDataSource(
        data=dict(
            x=pc[:,0],
            y=pc[:,1],
            desc=df3.index,
        )
    )

    #p = figure(title='PCA plot', plot_width=500, plot_height=400) 
    p = figure(title='PCA plot', plot_width=500, plot_height=400, tools=[hover]) 
    
    p.circle('x', 'y', source=source)
    #p.circle(x, y)

    p.xaxis.axis_label = "x"
    p.yaxis.axis_label = "y"

    figJS, figDiv = components(p)

    return(render_template(
        "figures.html",
        figJS=figJS,
        figDiv=figDiv))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        app.debug = True

    # Get port form environment or default to 5000
    PORT = int(os.getenv("PORT", 5000))
    # Check for env. variable or start on localhost
    INTERFACE = os.getenv("INTERFACE", "localhost")
    app.run(host=INTERFACE, port=PORT)
