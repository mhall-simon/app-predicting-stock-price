# Dash application for predicting stock prices
from dash import Dash, dcc, html, Input, Output
import os
import pandas as pd
import pandas_datareader.data as web
import plotly.express as px
from pycaret.time_series import TSForecastingExperiment


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.H2('Stock Price Predictor - ML Powered Predictions'),
    dcc.Dropdown(['GLW', 'AAPL', 'TSLA', 'META', 'AMZN'],
        'GLW',
        id='dropdown'
    ),
    dcc.Dropdown(['High','Low','Open','Close','Adj Close','Volume'],
        'Adj Close',
        id='price-graph-dropdown'
    ),
    html.Div(id='display-value'),
    dcc.Graph(id='price-graph'),
    html.Hr(),

    html.H3('Model Selection and Parameters'),
    dcc.Dropdown(['exp_smooth','lr_cds_dt','en_cds_dt','ridge_cds_dt','lasso_cds_dt','lar_cds_dt','llar_cds_dt',
    'br_cds_dt','huber_cds_dt'],
        'exp_smooth',
        id='model-dropdown'
    ),

    dcc.Graph(id='pycaret-graph'),

    dcc.Store(id='stock-data')
])

@app.callback(Output('display-value', 'children'),
                [Input('dropdown', 'value')])
def display_value(value):
    df = pd.DataFrame.from_records(value)
    return f'Selected Equity: {value}'


@app.callback(Output('price-graph', 'figure'),
                [Input('stock-data', 'data'),
                Input('price-graph-dropdown', 'value')])
def display_price_graph(value, col):
    df = pd.DataFrame.from_records(value)

    fig = px.line(df, x="Date", y=col, title=f'Equity Graph: {col}')
    return fig


@app.callback(Output('stock-data', 'data'),
                [Input('dropdown', 'value')])
def pull_stock_price(value):
    df = web.DataReader(value, 'yahoo')
    df = df.reset_index()

    return df.to_dict('records')


@app.callback(
    Output('pycaret-graph', 'figure'),
    [Input('stock-data', 'data'),
    Input('model-dropdown', 'value'),
    Input('price-graph-dropdown', 'value')]
)
def pycaret_graph(value, model, col):
    df = pd.DataFrame.from_records(value)
    df["Date"] = pd.to_datetime(df["Date"])

    df.set_index("Date")
    df = df[[col]]

    exp = TSForecastingExperiment()
    exp.setup(data=df, fh=90, fold=3, fig_kwargs={'renderer':'png'}, seasonal_period="D", transform_target='log')

    model = exp.create_model(model)
    final = exp.finalize_model(model)

    return exp.plot_model(final, plot="forecast", return_fig=True)

if __name__ == '__main__':
    app.run_server(debug=True)