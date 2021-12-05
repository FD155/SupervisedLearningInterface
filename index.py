from dash import *
from dash.dependencies import Input, Output
from pages.header import *
from pages.layout_acceuil import layout_acceuil
from app import app
import dash_core_components as dcc
import dash_html_components as html

#layout rendu par l'application
app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    navbar,
    html.Div(id='page_content'),
])

#callback pour mettre à jour les pages
@app.callback(Output('page_content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname=='/acceuil' or pathname=='/':
        return layout_acceuil


if __name__ == '__main__':
    app.run_server(debug=True)
