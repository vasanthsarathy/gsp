import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import json
import random


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.YETI, dbc.icons.FONT_AWESOME])
server = app.server #nee

LINE_BREAK = dbc.Row(html.Br())

# Top heading and logo
heading = [html.H2("GSP"),html.H5("Grounded Semantic Parser")]
HEADER_ROW = dbc.Row(dbc.Col(heading, width={'size':5, 'offset':1}, align="center"))

# DASH APP LAYOUT
app.layout = html.Div([LINE_BREAK,
                       HEADER_ROW,
                       LINE_BREAK
                       ])


############# CALLBACKS ############################


#########################

def main():
    server.run(debug=True, port=8010)

if __name__ == "__main__":
    app.run_server(debug=True)
