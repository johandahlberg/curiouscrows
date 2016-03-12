import logging
import argparse
import os

from flask import Flask, render_template
from flask.ext.bootstrap import Bootstrap

from bokeh.plotting import figure, ColumnDataSource
from bokeh.embed import components
from bokeh.charts import Bar
from bokeh.models import HoverTool, BoxZoomTool, ResetTool, TapTool, CustomJS

from sklearn.preprocessing import Imputer, scale
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import gzip
import operator

import flask

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
    list_regions = CustomJS(code="""
        var indexes = cb_obj.get('selected')['1d'].indices;
        var descs = [];
        for (index in indexes) {
            if (index > 1) {
                break;
            }
            descs.push(cb_obj.get('data').desc[indexes[index]]);
        }
        window.vue.$data.selected = cb_obj.get('data').desc[indexes[0]];
        window.vue.$data.regions = descs;
        window.cb_obj = cb_obj
    """)

    tools = [HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("desc", "@desc"),
        ]
    ), BoxZoomTool(), ResetTool(), TapTool(callback=list_regions)]

    source = ColumnDataSource(
        data=dict(
            x=pc[:, 0],
            y=pc[:, 1],
            desc=df3.index,
            color=['navy'] * len(df3.index)
        )
    )

    show_region = CustomJS(args=dict(source=source), code="""
        var region = cb_obj.get('value');
        var data = source.get('data');
        var region_idx = data['desc'].indexOf(region);
        data['colors'][region_idx] = 'firebrick';
        source.trigger('change');
    """)

    plot = figure(title='PCA plot', plot_width=960, plot_height=600,
                  tools=tools, responsive=True)

    plot.background_fill_color = 'beige'
    plot.background_fill_alpha = 0.5
    plot.border_fill = '#FCFCFC'

    plot.circle('x', 'y', size=10, source=source, color='color',
                # set visual properties for selected glyphs
                # set visual properties for non-selected glyphs
                nonselection_fill_alpha=0.2)

    plot.xaxis.axis_label = "x"
    plot.yaxis.axis_label = "y"

    pca_fig_js, pca_fig_div = components(plot)
    all_regions = [region.decode('utf-8') for region in df3.index]
    return pca_fig_js, pca_fig_div, all_regions


def top_kpis_as_dict(municipality):
    rks = kpi_ranks(df3, municipality, missing)
    sorted_ranks = sorted(rks.items(), key=operator.itemgetter(1), reverse=True)
    top_kpi = [k[0] for k in sorted_ranks[:10]]

    selected = original_data[(original_data.kpi.isin(top_kpi)) & (original_data.municipality_name == municipality)]
    selected.value = selected.value.astype(float)

    return selected.to_json(orient='records')


def create_bar_plot():
    municipality = "Uppsala"

    rks = kpi_ranks(df3, municipality, missing)
    sorted_ranks = sorted(rks.items(), key=operator.itemgetter(1), reverse=True)
    top_kpi = [k[0] for k in sorted_ranks[:10]]

    selected = original_data[(original_data.kpi.isin(top_kpi)) & (original_data.municipality_name == municipality)]
    selected.value = selected.value.astype(float)

    bar_plot = Bar(
        selected,
        'municipality_name',
        values='value',
        group='kpi',
        title="Test bar plot")

    bar_fig_js, bar_fig_div = components(bar_plot)
    return bar_fig_js, bar_fig_div


# def construct_municip_to_identifier(data):
#     m2i = {}
#     for row in data.iterrows():
#         print row.values[2]
#         m2i[row[2]] = row[3]
#     return m2i


original_data = load_data()
df3, pc, missing = compute_principal_components(original_data)
# muncip2identifier = construct_municip_to_identifier(original_data )
#
# print(muncip2identifier)

def diff_kpi_ranks(df, kommun1, kommun2, missing):
    rankdiffs = {}
    for kpi in df.columns:
        if(missing[kpi][kommun1]==True): continue
        if(missing[kpi][kommun2]==True): continue
        all_m = sorted(df[kpi])
        val1 = df[kpi][kommun1]
        val2 = df[kpi][kommun2]
        rank1 = all_m.index(val1)
        rank2 = all_m.index(val2)
        rankdiffs[kpi]=(rank1-rank2)
    return(rankdiffs)

@app.route('/diff_kpis/<municip1>/<municip2>')
def diff_kpis(municip1, municip2):
    municip1 = municip1.encode('utf-8')
    municip2 = municip2.encode('utf-8')

    rd = diff_kpi_ranks(df3, municip1.encode('utf-8'), municip2.encode('utf-8'), missing)
    sorted_diffs = sorted(rd.items(), key=operator.itemgetter(1), reverse=True)
    top_diffs = map(lambda x: x[0], sorted_diffs[:10])
    bottom_diffs = map(lambda x: x[0], sorted_diffs[-10:])
    selected = original_data[
        (original_data.kpi.isin(top_diffs) | original_data.kpi.isin(bottom_diffs)) &
        ((original_data.municipality_name == municip1) | (original_data.municipality_name == municip2))]
    sl = selected.pivot(index='municipality_name', columns='kpi_desc', values='value').T
    sl["kpi_desc"] = sl.index
    sl.columns = ['first', 'second']
    return sl.to_json(orient='records')

# @app.route('/municip_to_identifier/<name>')
# def municip_to_identifier(name):
#     original_data


@app.route('/top_kpis/<municipality>')
def top_kpis(municipality):
    return top_kpis_as_dict(municipality.encode('utf-8'))


@app.route('/')
def index():
    pca_fig_js, pca_fig_div, all_regions = create_pca_plot()

    return render_template("figures.html",
                           pcaFigJs=pca_fig_js,
                           pcaFigDiv=pca_fig_div,
                           all_regions=all_regions)


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
