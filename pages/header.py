import dash_bootstrap_components as dbc
from dash import *
from PIL import Image
import os
import dash_html_components as html
import dash
import pathlib

os.chdir(str(pathlib.Path(__file__).parent.parent)+"/assets")

im = Image.open('logo.png')

navbar = dbc.Navbar(
    [
        html.A(
            # Alignement vertical de l'image et de l'acceuil
            dbc.Row(
                [   #logo
                    dbc.Col(html.Img(src=im, height="40px")),
                    #Navlink Acceuil
                   dbc.Col( dbc.NavLink("Acceuil", href="/acceuil",style={'color':'white'})),
                ],
                align="cesnter", className="g-0",
            ),
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
    ],
    color="dark",
    dark=True,
)


