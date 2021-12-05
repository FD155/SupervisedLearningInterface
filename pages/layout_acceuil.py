import dash_bootstrap_components as dbc
from PIL import Image
import pandas as pd
import io 
import base64
from dash import *
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import callbacks
from app import *
import pathlib
import os

os.chdir(str(pathlib.Path(__file__).parent.parent)+"/assets")

im = Image.open('logo.png')
html.Img(src=im, height="40px")

body = dbc.Container([
        html.Br(),
        dbc.Row(
                [
                dbc.Col(

                    html.Div(
                        [   html.Br([]),
                            html.H5("Bienvenue!",style={'color':'darkblue','backgroundColor':'white'}),
                            html.Br([]),
                            html.P(
                                "\
                            Vous êtes sur la page d'acceuil ! \
                            Nous vous prsentons ici une application destinée au Machine Learning",

                                style={"color": "#000406"},

                            ),
                            html.P(
                                "\
                            Choisissez vos données.\
                        Vous pouvez accéder au dashboard via la barre de navigation ou en cliquant directement ci-dessous.",


                                style={"color": "#000406"},

                            ),
                             
                            dbc.Row(
                                [
                                    dcc.Upload(
                                                id='upload',
                                                children=html.Div([
                                                    'Insérez ou ',
                                                    html.A('selectionnez un fichier')
                                                ]),
                                                style={
                                                    'color':'black',
                                                    'backgroundColor':'lightgray',
                                                    'width': '150%',
                                                    'height': '60px',
                                                    'lineHeight': '60px',
                                                    'borderWidth': '1px',
                                                    'borderStyle': 'dashed',
                                                    'borderRadius': '5px',
                                                    'textAlign': 'center',
                                                    'margin': '10px'
                                                }
                                            ) ]),
                            
                        ]

                         ),style={'color':'black','backgroundColor':'white'})], justify="center", align="center"
                    ),
            
            
            
                        html.Br([]),
                        html.Div(id='json_info', style={'display':'none'}),
                        html.Div(id='info_output'),
                           
                        html.Br([]),                    
                        html.Div(id='json_df', style={'display':'none'}),
                        html.Div(id='df_output'),
                       
                        html.Br([]),
                        html.Div(id='Stat_output'),
                        html.Br([]),  




                        html.Br([]),
                        html.Div(id='Var_cible'),
                        html.Br([]),  

                        html.Div(id='Var_desc'), 
                        html.Br([]),  


                        html.Div(id='Var_type'), 
                        html.Br([]),  

                        html.Div(id='param'),                       

                        html.Br([]),  

                        html.Div(id='submit'),   
                        
                        
                        html.Div(id='ML'),  


                        html.Br([]),  
                        html.Div( dbc.Navbar(
                            [
                              dbc.Row(
                                    [
                                        dbc.Col(dbc.NavLink("M2 SISE - Université Lyon 2 ", href="https://www.univ-lyon2.fr/master-2-informatique-statistique-et-informatique-sise-1",
                                                            style={'color':'white'}),width={"size": 6}),
                                         dbc.Col(html.Span('Auteurs : Afaf BEN HAJ, Franck DORONZO et Marie VACHET  ', style={'color':'#ffff'}), 
                                        width={"size": 6}),
                                    ],  className="g-0", justify="end"
                                    
                                    ),
                                dbc.NavbarToggler(id="navbar-toggler2"),
                            ],
                            color="dark",
                            dark=True, 
                        )), 
     html.Br(),
],style={"height": "100vh"}
)





layout_acceuil =  html.Div([body],style={'background-image': 'url("/accueil.jpg")'})
