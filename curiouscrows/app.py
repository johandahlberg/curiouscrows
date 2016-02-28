import random
import argparse
import os

from flask import Flask, render_template

from bokeh.plotting import figure
from bokeh.embed import components

app = Flask(__name__)


@app.route('/')
def index():

    y = [random.random() for n in range(50)]
    x = [random.random() for n in range(50)]

    p = figure(title='A Bokeh plot',
               plot_width=500,
               plot_height=400)
    p.circle(x, y)

    p.xaxis.axis_label = "x"
    p.yaxis.axis_label = "y"

    figJS, figDiv = components(p)

    return(render_template(
        "figures.html",
        y=y,
        figJS=figJS,
        figDiv=figDiv))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        app.debug = True

    # Get port form environment or default to 5000
    PORT = int(os.getenv('PORT', 5000))
    app.run(port=PORT)
