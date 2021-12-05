from dash.dependencies import Input, Output, State
from app import *
import dash_table
import pandas as pd
import io 
import base64
from dash import *
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from functions import regressionLog,adl, knnClass, knn_reg,decTree, RegLineaire,svr_reg


@app.callback(Output('json_info', 'children'),
              [Input('upload', 'contents'),
               Input('upload', 'filename')])
def load_df(contents, filename):
   if (contents is None):
       df = pd.DataFrame()
   elif contents:
        # Modify the read_csv callback part
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            input_data  = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            info_dataframe = pd.DataFrame(data={
                                              "Type de donnée": input_data.dtypes,
                                              "Nombre de valeurs manquantes": input_data.isna().sum(),
                                              "Nombre de valeurs uniques": input_data.nunique()
                                                   })
            # adding index as a row
            info_dataframe.reset_index(level=0, inplace=True)
            info_dataframe.rename(columns={'index':'Nom de colonne'}, inplace=True)
            info_dataframe['Type de donnée'] = info_dataframe['Type de donnée'].astype(str)
            return info_dataframe.to_json(date_format='iso', orient='split')
            
        else:
            #print(e)
            return pd.DataFrame.style.set_properties(**{'background-color': 'black'})(data={'Error': 'Importez un fichier csv'}, index=[0]).to_json(date_format='iso', orient='split')


@app.callback(Output('json_df', 'children'),
              [Input('upload', 'contents'),
               Input('upload', 'filename')])
def load_df2(contents, filename):
   if (contents is None):
       df = pd.DataFrame()
   elif contents:
        # Modify the read_csv callback part
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'csv' in filename:
                # Assume that the user uploaded a CSV file
            data  = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            if (len(data.columns)==1):
                return pd.DataFrame(data={'Error': 'Le jeu de données doit être délimité par des virgules'}, index=[0]).to_json(date_format='iso', orient='split')
            else:
                return data.to_json(date_format='iso', orient='split')
        else:
            #print(e)
            return pd.DataFrame(data={'Error': 'Importez un fichier csv'}, index=[0]).to_json(date_format='iso', orient='split')
        

# callback to take and output the uploaded file
@app.callback(Output('info_output', 'children'),
              [Input('json_info', 'children')]) 
def update_output(json_info):
   if (json_info is None):
       df = pd.DataFrame()
       child = html.Div([])
   else:
       df = pd.read_json(json_info, orient='split')
       child = html.Div([
                   html.P("Informations des données : ",style={'color':'darkblue','backgroundColor':'white'}),
                   dash_table.DataTable(
                       id='table',     
                       data= df.to_dict("records"),
                       columns=[{"name": i, "id": i} for i in df.columns],
                       editable=False,
                       #n_fixed_columns=2,
                       style_table={'maxWidth': '1500px'},
                       style_cell = {"fontFamily": "Arial", "size": 10, 'textAlign': 'center', 'color':'dark'},
                       css=[{'selector': '.dash-cell div.dash-cell-value', 'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'}]
                       )])
   return child


# callback to take and output the uploaded file
@app.callback(Output('df_output', 'children'),
              [Input('json_df', 'children')]) 
def update_output2(json_df):
   if (json_df is None):
       df = pd.DataFrame()
       child = html.Div([])
   else:
       df = pd.read_json(json_df, orient='split')
       child = html.Div([
                   html.P("Aperçu des données : ",style={'color':'darkblue','backgroundColor':'white'}),
                   dash_table.DataTable(
                       id='table',     
                       data= df.head(5).to_dict("records"),
                       columns=[{"name": i, "id": i} for i in df.columns],
                       editable=False,
                       #n_fixed_columns=2,
                       style_table={'maxWidth': '1500px'},
                       style_cell = {"fontFamily": "Arial", "size": 10, 'textAlign': 'center', 'color':'dark'},
                       css=[{'selector': '.dash-cell div.dash-cell-value', 'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'}]
                       )])
   return child


# callback to take and output the uploaded file
@app.callback(Output('Stat_output', 'children'),
              [Input('json_df', 'children')]) 
def update_output3(json_df):    
   if (json_df is None):
       df = pd.DataFrame()
       child = html.Div([])
   else:
       df = pd.read_json(json_df, orient='split')
       df2 = pd.DataFrame(df.describe().round(2).reset_index())
       child = html.Div([
                   html.P("Description statistiques des données : ",style={'color':'darkblue','backgroundColor':'white'}),
                   dash_table.DataTable(
                       id='table',     
                       data= df2.to_dict("records"),
                       columns=[{"name": i, "id": i} for i in df2.columns],
                       editable=False,
                       #n_fixed_columns=2,
                       style_table={'maxWidth': '1500px'},
                       style_cell = {"fontFamily": "Arial", "size": 10, 'textAlign': 'center', 'color':'dark'},
                       css=[{'selector': '.dash-cell div.dash-cell-value', 'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'}]
                       )])
   return child


# callback to take and output the uploaded file
@app.callback(Output('Var_cible', 'children'),
              [Input('json_df', 'children')]) 
def update_output4(json_df):    
   if (json_df is None):
       df = pd.DataFrame()
       child = html.Div([])
   else:
       df = pd.read_json(json_df, orient='split')
       my_list = df.columns.values.tolist()
       
       child = html.Div([
                        html.H5("PARTIE MACHINE LEARNING ",style={  'width': '100%','lineHeight': '50px', 
                        'height': '50px','textAlign': 'center','color':'white','backgroundColor':'#204e48'}),
                         html.P("Choisissez votre variable cible : ",style={'color':'darkblue','backgroundColor':'white'}),
                       dcc.Dropdown(
                                id='cible_dropdown',
                                options=[{'label': x, 'value': x} for x in my_list],
                                value=my_list[0])])
   return child


@app.callback(
    Output('Var_desc', 'children'),
    Input('cible_dropdown', 'value'),
    State('json_df', 'children')
)
def update_output5(value,json_df):
    if (json_df is None):
       df = pd.DataFrame()
       child = html.Div([])
    else:
        df = pd.read_json(json_df, orient='split')
        my_list = df.columns.values.tolist()
        del my_list[my_list.index(value)]
        
        child = html.Div([
                           html.P("Choisissez vos variables descriptives : ",style={'color':'darkblue','backgroundColor':'white'}),
                           dcc.Dropdown(
                                     id='desc_dropdown',
                                     options=[{'label': x, 'value': x} for x in my_list],
                                     value=my_list,
                                     multi=True),

                           ])       
    return child



@app.callback(
    Output('Var_type', 'children'),
    Input('cible_dropdown', 'value'),
    State('json_df', 'children'),
)
def update_output7(value,json_df):
    if (json_df is None):
        df = pd.DataFrame()
        child = html.Div([])       
    else:
        df = pd.read_json(json_df, orient='split')
        valueType = df[value].dtypes  
       
    if (valueType == object):
        child = html.Div([  'La variable cible est de type "{}"'.format(valueType),html.Br([]),  html.Br([]),
                            html.P( "Vous pouvez appliquer les algorithmes de Machine Learning suivants : ", style={"color": "#000406"},),
                            dbc.Button('Regression Logistique', id='BRegLog',
                                       outline=True, color="success" ,style=dict(marginRight=20) , n_clicks=0),
                            dbc.Button('Analyse Discriminante Linéaire', id='BADL', 
                                       outline=True, color="success", style=dict(marginRight=20) , n_clicks=0),
                            dbc.Button('KNN Classification', id='BknnC', 
                                       outline=True, color="success", style=dict(marginRight=20) , n_clicks=0),
                            
                            dbc.Button('KNN Regression', id='BknnR',
                                       outline=True, color="success",style=dict(marginRight=20,display='none'), n_clicks=0),
                            dbc.Button('Arbre de déscision', id='Btree',
                                       outline=True, color="success",style=dict(marginRight=20,display='none'), n_clicks=0),
                            dbc.Button('Regression Linéaire', id='BRegLin', 
                                       outline=True, color="success",style=dict(marginRight=20,display='none'), n_clicks=0),
                            dbc.Button('SVR Regression', id='BSVR',
                                        outline=True, color="success",style=dict(marginRight=20,display='none'), n_clicks=0), html.Br([]),
                        ])
    elif (valueType == float or valueType == int): 
        child = html.Div([
                            'La variable cible est de type "{}"'.format(valueType),html.Br([]), html.Br([]) ,  
                            html.P( "Vous pouvez appliquer les algorithmes de Machine Learning suivants : ", style={"color": "#000406"},),
                                                        dbc.Button('Regression Logistique', id='BRegLog',
                                       outline=True, color="success" ,style=dict(marginRight=20,display='none') , n_clicks=0),
                            dbc.Button('Analyse Discriminante Linéaire', id='BADL', 
                                       outline=True, color="success", style=dict(marginRight=20,display='none') , n_clicks=0),
                            dbc.Button('KNN Classification', id='BknnC', 
                                       outline=True, color="success", style=dict(marginRight=20,display='none') , n_clicks=0),
                            
                            dbc.Button('KNN Regression', id='BknnR',
                                       outline=True, color="success",style=dict(marginRight=20), n_clicks=0),
                            dbc.Button('Arbre de déscision', id='Btree',
                                       outline=True, color="success",style=dict(marginRight=20), n_clicks=0),
                            dbc.Button('Regression Linéaire', id='BRegLin', 
                                       outline=True, color="success",style=dict(marginRight=20), n_clicks=0),
                            dbc.Button('SVR Regression', id='BSVR',
                                        outline=True, color  ="success",style=dict(marginRight=20), n_clicks=0), html.Br([]),
                        ])   
        
    child2 = html.Div([
        dbc.Label("Souhaitez vous personnaliser vos paramètres d'algorithmes"),
        dbc.Checklist(options=[{"label": "Oui", "value":1},],
                                                    value=[],
                                                    id="param_switch",
                                                    inline=True,
                                                    switch=True,
                                                ),     html.Br([]),
        
        dbc.Button('Lancer l\'algorithme', id='go',
                                        outline=True, color  ="success",style=dict(marginRight=20,textAlign='center'), n_clicks=0), html.Br([]),])

    return  html.Div([child,child2])    
    

# -------------------- GLOBAL VARIABLE FOR ML -----------------------
choisis = False
algo = '' 

@app.callback(
    Output('param', 'style'),
    Input('param_switch', 'value'),
)
def update_output9(value):
    global choisis
    if value==[1]:
        choisis = True
        return {'display': 'block'}
    else:
        choisis = False
        return {'display': 'none'}
    

@app.callback(
    Output('param', 'children'),
    [Input('BRegLog', 'n_clicks'),   
     Input('BADL', 'n_clicks'),
     Input('BknnC', 'n_clicks'),
     Input('BknnR', 'n_clicks'),
     Input('Btree', 'n_clicks'),
     Input('BRegLin', 'n_clicks'),
     Input('BSVR', 'n_clicks')],
)
def update_output8(B1,B2,B3,B4,B5,B6,B7):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    global algo
    if 'BRegLog' in changed_id:
        child = html.Div([
                        dbc.Label("max_iter"),
                        dbc.Input(placeholder="Insérez un entier...",id="select1", value=100, type='number', min=1,step=1, className="mb-3"),   
                        dbc.Label("l1_ratio"),
                        dbc.Input(placeholder="Insérez un réel entre 0 et 1...", value=0.5, type='number', min=0, max=1, step= 0.05, id="select2", className="mb-3"), 
                        dbc.Input(id="select3",style={'display':'none'}),     dbc.Input(id="select4",style={'display':'none'}),                             
                ])   
        algo = 'BRegLog'        
    elif 'BADL' in changed_id:
        child = html.Div([
                        dbc.Label("solver"),
                        dbc.Select(
                                id="select1",
                                options=[
                                    {"label": "svd", "value": "svd"},
                                    {"label": "lsqr", "value": "lsqr"},
                                    {"label": "eigen", "value": "eigen"},
                                ],value=['svd'],
                            ), 
                        dbc.Input(id="select2",style={'display':'none'}),                          
                        dbc.Input(id="select3",style={'display':'none'}),     
                        dbc.Input(id="select4",style={'display':'none'}),                                     
                ])
        algo = 'BADL' 
        
    elif 'BknnC' in changed_id:
        child = html.Div([
                        dbc.Label("n_neighbors"),
                        dbc.Input(placeholder="Insérez un entier supérieur à 0....", id="select1",   value=1, type='number', min=1,step=1,  className="mb-3"), 
                        dbc.Input(id="select2",style={'display':'none'}), dbc.Input(id="select3",style={'display':'none'}),     dbc.Input(id="select4",style={'display':'none'}),   
                        ])
        algo = 'BknnC'  
       
    elif 'BknnR' in changed_id:
        child = html.Div([
                        dbc.Label("n_neighbors"),
                        dbc.Input(placeholder="Insérez un entier supérieur à 0....", id="select1", value=1, type='number', min=1,step=1,  className="mb-3"), 
                        dbc.Input(id="select2",style={'display':'none'}), dbc.Input(id="select3",style={'display':'none'}),     dbc.Input(id="select4",style={'display':'none'}),    
                        ])
        algo = 'BknnR'
                    
    elif 'Btree' in changed_id:
        child = html.Div([
                        dbc.Label("max_depth"),
                        dbc.Input(placeholder="Insérez un entier supérieur à 0....", id="select1",value=1, type='number', min=1,step=1, className="mb-3"),
                        dbc.Label("min_samples_leaf"),
                        dbc.Input(placeholder="Insérez un réel supérieur à 0....",  id="select2", value=1, type='number', min=1,step=1,className="mb-3"), 
                        dbc.Label("splitter"),
                        dbc.Select(
                                id="select3",
                                options=[
                                    {"label": "best", "value": "best"},
                                    {"label": "random", "value": "random"},
                                ],value=['best'],
                            ),                  
                        dbc.Label("min_samples_split"),
                        dbc.Input(placeholder="Insérez un réel supérieur à 0.....", id="select4", value=1, type='number', min=1,step=1, className="mb-3"),                   
                ])
        algo = 'Btree' 

    elif 'BRegLin' in changed_id:
        child = html.Div([
                        dbc.Label("alpha"),
                        dbc.Input(placeholder="Insérez un réel supérieur à 0....", id="select1",value=1, type='number', min=1,step=1, className="mb-3"),
                        dbc.Label("l1_ratio"),
                        dbc.Input(placeholder="Insérez un réel entre 0 et 1...", id="select2", value=0.5, type='number', min=0, max=1, step= 0.05,className="mb-3"),  
                        dbc.Input(id="select3",style={'display':'none'}),     dbc.Input(id="select4",style={'display':'none'}),   
                ])
        algo = 'BRegLin' 

    elif 'BSVR' in changed_id:
        child = html.Div([
                        dbc.Label("kernel"),
                        dbc.Select(
                                id="select1",
                                options=[
                                    {"label": "rbf", "value": "rbf"},
                                    {"label": "linear", "value": "linear"},
                                    {"label": "sigmoid", "value": "sigmoid"},
                                    {"label": "precomputed", "value": "precomputed"},
                                ],value=['rbf'],
                            ),
                        
                        dbc.Label("gamma"),
                        dbc.Select(
                                id="select2",
                                options=[
                                    {"label": "scale", "value": "scale"},
                                    {"label": "auto", "value": "auto"},
                                ],value=['scale'],
                            ), 
                        dbc.Label("C"),
                        dbc.Input(placeholder="Insérez un réel...", id="select3",  value=0.5, type='number', min=0,  step= 0.05,className="mb-3"),
                        dbc.Input(id="select4",style={'display':'none'}),   
                      ]) 
        
        algo = 'BSVR' 

    return  child

# -------------------- GLOBAL VARIABLE FOR ML -----------------------
l1_ratio=''
max_iter=''
solver=''
n_neighbors=''
splitter=''
max_depth=''
min_samples_leaf=''
min_samples_split=''
alpha=''
l1_ratio=''
kernel=''
gamma=''
C=''

@app.callback(
    Output('paramFunct', 'children'),
    [Input('select1', 'value'),   
     Input('select2', 'value'), 
     Input('select3', 'value'), 
     Input('select4', 'value'), 
     ], 
)
def update_output11(v1,v2,v3,v4):
    global algo
    global l1_ratio
    global max_iter
    global solver
    global n_neighbors
    global splitter
    global max_depth
    global min_samples_leaf
    global min_samples_split
    global alpha
    global l1_ratio
    global kernel
    global gamma
    global C
    if algo=='BRegLog':
        max_iter=v1
        l1_ratio=v2
    if algo=='BADL':
        solver=v1
    if algo=='BknnC':
        n_neighbors=v1   
    if algo=='BknnR':
        n_neighbors=v1
    if algo=='Btree':
        max_depth=v1
        min_samples_leaf=v2
        splitter=v3
        min_samples_split=v4
    if algo=='BRegLin':
        alpha=v1
        l1_ratio=v2
    if algo=='BSVR':
        kernel=v1
        gamma=v2
        C=v3
    return html.Div([])


@app.callback(
    Output('ML', 'children'),
    Input('go', 'n_clicks'),    
    [State('cible_dropdown', 'value'),
     State('desc_dropdown', 'value'), 
    State('json_df', 'children'),],
)
def update_output10(B1,valueC,valueD,json_df):
    childML = html.Div([])

    if (json_df is None):
       df = pd.DataFrame()   
    else:
       df = pd.read_json(json_df, orient='split')
       X =  df[valueD]
       y = df[valueC]
   
    global choisis 
    global algo
    global l1_ratio
    global max_iter
    global solver
    global n_neighbors
    global splitter
    global max_depth
    global min_samples_leaf
    global min_samples_split
    global alpha
    global l1_ratio
    global kernel
    global gamma
    global C 
    
    if choisis==False:
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]
        if 'go' in changed_id:
            if algo=='BRegLog':
                a = regressionLog(X,y,choisis=choisis) 
                fig = go.Figure(data=go.Scatter(y=a["prediction"],
                                        mode='lines+markers',
                                        name='Prédictions'))
                childML = html.Div([
                                html.H4("Indicateurs pour la Régression logistique sur les données", style={'fontWeight':'bold'}),
                                html.P("f1_score ", style={'fontWeight':'bold'}),
                                html.Div(round(a["f1_score"], 2), ),
                                html.P("precision ", style={'fontWeight':'bold'}),
                                html.Div(round(a["precision"], 2),),
                                html.P("rappel ", style={'fontWeight':'bold'}),
                                html.Div(round(a["rappel"], 2),),
                                html.P("Temps d'exécution de l'algorithme en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(round(a["temps"], 2),),
                                dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                ])
                
            if algo=='BADL':
                a = adl(X,y,choisis=choisis)
                fig = go.Figure(data=go.Scatter(y=a["prediction"],
                                        mode='lines+markers',
                                        name='Prédictions'))
                childML = html.Div([
                                html.H4("Indicateurs pour l'Analyse Discriminante Linéaire sur les données", style={'fontWeight':'bold'}),
                                html.P("f1_score ", style={'fontWeight':'bold'}),
                                html.Div(round(a["f1_score"], 2), ),
                                html.P("precision ", style={'fontWeight':'bold'}),
                                html.Div(round(a["precision"], 2),),
                                html.P("rappel ", style={'fontWeight':'bold'}),
                                html.Div(round(a["rappel"], 2),),
                                html.P("Temps d'exécution de l'algorithme en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(round(a["temps"], 2),),
                                dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                ])
                
            if algo=='BknnC':
                a = knnClass(X,y,choisis=choisis)   
                fig = go.Figure(data=go.Scatter(y=a["prediction"],
                                        mode='lines+markers',
                                        name='Prédictions'))
                childML = html.Div([
                                html.H4("Indicateurs pour la Classification des K plus proches voisins sur les données", style={'fontWeight':'bold'}),
                                html.P("f1_score ", style={'fontWeight':'bold'}),
                                html.Div(round(a["f1_score"], 2), ),
                                html.P("precision ", style={'fontWeight':'bold'}),
                                html.Div(round(a["precision"], 2),),
                                html.P("rappel ", style={'fontWeight':'bold'}),
                                html.Div(round(a["rappel"], 2),),
                                html.P("Temps d'exécution de l'algorithme en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(round(a["temps"], 2),),
                                dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                 ])
                
            if algo=='BknnR':
                a = knn_reg(X,y,choisis=choisis)
                fig = go.Figure(data=go.Scatter(y=a["prediction"],
                                        mode='lines+markers',
                                        name='Prédictions'))
                childML = html.Div([
                                html.H4("Indicateurs pour la Regréssion des K plus proches voisins sur les données", style={'fontWeight':'bold'}),
                                html.P("R2 ", style={'fontWeight':'bold'}),
                                html.Div(round(a["R2"], 2),),
                                html.P("mse ", style={'fontWeight':'bold'}),
                                html.Div(round(a["mse"], 2), ),
                                html.P("Temps d'exécution de l'algorithme en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(round(a["temps"], 2),),
                                dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                ])
                
            if algo=='Btree':
                a = decTree(X,y,choisis=choisis)
                fig = go.Figure(data=go.Scatter(y=a["prediction"],
                                        mode='lines+markers',
                                        name='Prédictions'))
                childML = html.Div([
                                html.H4("Indicateurs pour l'Arbre de décision sur les données", style={'fontWeight':'bold'}),
                                html.P("R2 ", style={'fontWeight':'bold'}),
                                html.Div(round(a["R2"], 2),),
                                html.P("mse ", style={'fontWeight':'bold'}),
                                html.Div(round(a["mse"], 2), ),
                                html.P("Temps d'exécution de l'algorithme en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(round(a["temps"], 2),),
                                dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                 ])
                
            if algo=='BRegLin':
                a = RegLineaire(X,y,choisis=choisis)
                fig = go.Figure(data=go.Scatter(y=a["prediction"],
                                        mode='lines+markers',
                                        name='Prédictions'))
                childML = html.Div([
                                html.H4("Indicateurs pour la Régression linéaire sur les données ", style={'fontWeight':'bold'}),
                                html.P("R2 ", style={'fontWeight':'bold'}),
                                html.Div(round(a["R2"], 2),),
                                html.P("mse ", style={'fontWeight':'bold'}),
                                html.Div(round(a["mse"], 2), ),
                                html.P("Temps d'exécution de l'algorithme en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(round(a["temps"], 2),),
                                dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                 ])
                
            if algo=='BSVR':
                a = svr_reg(X,y,choisis=choisis) 
                fig = go.Figure(data=go.Scatter(y=a["prediction"],
                                        mode='lines+markers',
                                        name='Prédictions'))
                childML = html.Div([
                                html.H4("Indicateurs SVR sur les données", style={'fontWeight':'bold'}),
                                html.P("R2 ", style={'fontWeight':'bold'}),
                                html.Div(round(a["R2"], 2),),
                                html.P("precision ", style={'fontWeight':'bold'}),
                                html.Div(round(a["precision"], 2), ),
                                html.P("mse ", style={'fontWeight':'bold'}),
                                html.Div(round(a["mse"], 2), ),
                                html.P("Temps d'exécution de l'algorithme en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(round(a["temps"], 2),),
                                dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                 ])
           
            return childML
    
    elif choisis==True:
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]
        if 'go' in changed_id:
            if algo=='BRegLog':
                a = regressionLog(X,y,l1_ratio=l1_ratio,max_iter=max_iter,choisis=choisis) 
                fig = go.Figure(data=go.Scatter(y=a["prediction"],
                                        mode='lines+markers',
                                        name='Prédictions'))
                childML = html.Div([
                                html.H4("Indicateurs pour la Régression logistique sur les données", style={'fontWeight':'bold'}),
                                html.P("f1_score ", style={'fontWeight':'bold'}),
                                html.Div(round(a["f1_score"], 2), ),
                                html.P("precision ", style={'fontWeight':'bold'}),
                                html.Div(round(a["precision"], 2),),
                                html.P("rappel ", style={'fontWeight':'bold'}),
                                html.Div(round(a["rappel"], 2),),
                                html.P("Temps d'exécution de l'algorithme en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(round(a["temps"], 2),),
                                dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                ])
                
            if algo=='BADL':
                a = adl(X,y,solver=solver,choisis=choisis)
                fig = go.Figure(data=go.Scatter(y=a["prediction"],
                                        mode='lines+markers',
                                        name='Prédictions'))
                childML = html.Div([
                                html.H4("Indicateurs pour l'Analyse Discriminante Linéaire sur les données", style={'fontWeight':'bold'}),
                                html.P("f1_score ", style={'fontWeight':'bold'}),
                                html.Div(round(a["f1_score"], 2), ),
                                html.P("precision ", style={'fontWeight':'bold'}),
                                html.Div(round(a["precision"], 2),),
                                html.P("rappel ", style={'fontWeight':'bold'}),
                                html.Div(round(a["rappel"], 2),),
                                html.P("Temps d'exécution de l'algorithme en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(round(a["temps"], 2),),
                                dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                ])
                
            if algo=='BknnC':
                a = knnClass(X,y,n_neighbors=n_neighbors,choisis=choisis)   
                fig = go.Figure(data=go.Scatter(y=a["prediction"],
                                        mode='lines+markers',
                                        name='Prédictions'))
                childML = html.Div([
                                html.H4("Indicateurs pour la Classification des K plus proches voisins sur les données", style={'fontWeight':'bold'}),
                                html.P("f1_score ", style={'fontWeight':'bold'}),
                                html.Div(round(a["f1_score"], 2), ),
                                html.P("precision ", style={'fontWeight':'bold'}),
                                html.Div(round(a["precision"], 2),),
                                html.P("rappel ", style={'fontWeight':'bold'}),
                                html.Div(round(a["rappel"], 2),),
                                html.P("Temps d'exécution de l'algorithme en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(round(a["temps"], 2),),
                                dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                 ])
                
            if algo=='BknnR':
                a = knn_reg(X,y,n_neighbors=n_neighbors,choisis=choisis)
                fig = go.Figure(data=go.Scatter(y=a["prediction"],
                                        mode='lines+markers',
                                        name='Prédictions'))
                childML = html.Div([
                                html.H4("Indicateurs pour la Regréssion des K plus proches voisins sur les données", style={'fontWeight':'bold'}),
                                html.P("R2 ", style={'fontWeight':'bold'}),
                                html.Div(round(a["R2"], 2),),
                                html.P("mse ", style={'fontWeight':'bold'}),
                                html.Div(round(a["mse"], 2), ),
                                html.P("Temps d'exécution de l'algorithme en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(round(a["temps"], 2),),
                                dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                ])
                
            if algo=='Btree':
                a = decTree(X,y,splitter=splitter,max_depth=max_depth,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,choisis=choisis)
                fig = go.Figure(data=go.Scatter(y=a["prediction"],
                                        mode='lines+markers',
                                        name='Prédictions'))
                childML = html.Div([
                                html.H4("Indicateurs pour l'Arbre de décision sur les données", style={'fontWeight':'bold'}),
                                html.P("R2 ", style={'fontWeight':'bold'}),
                                html.Div(round(a["R2"], 2),),
                                html.P("mse ", style={'fontWeight':'bold'}),
                                html.Div(round(a["mse"], 2), ),
                                html.P("Temps d'exécution de l'algorithme en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(round(a["temps"], 2),),
                                dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                 ])
                
            if algo=='BRegLin':
                a = RegLineaire(X,y,alpha=alpha,l1_ratio=l1_ratio,choisis=choisis)
                fig = go.Figure(data=go.Scatter(y=a["prediction"],
                                        mode='lines+markers',
                                        name='Prédictions'))
                childML = html.Div([
                                html.H4("Indicateurs pour la Régression linéaire sur les données ", style={'fontWeight':'bold'}),
                                html.P("R2 ", style={'fontWeight':'bold'}),
                                html.Div(round(a["R2"], 2),),
                                html.P("mse ", style={'fontWeight':'bold'}),
                                html.Div(round(a["mse"], 2), ),
                                html.P("Temps d'exécution de l'algorithme en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(round(a["temps"], 2),),
                                dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                 ])
                
            if algo=='BSVR':
                a = svr_reg(X,y, kernel=kernel,gamma=gamma,C=C, choisis=choisis) 
                fig = go.Figure(data=go.Scatter(y=a["prediction"],
                                        mode='lines+markers',
                                        name='Prédictions'))
                childML = html.Div([
                                html.H4("Indicateurs SVR sur les données", style={'fontWeight':'bold'}),
                                html.P("R2 ", style={'fontWeight':'bold'}),
                                html.Div(round(a["R2"], 2),),
                                html.P("precision ", style={'fontWeight':'bold'}),
                                html.Div(round(a["precision"], 2), ),
                                html.P("mse ", style={'fontWeight':'bold'}),
                                html.Div(round(a["mse"], 2), ),
                                html.P("Temps d'exécution de l'algorithme en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(round(a["temps"], 2),),
                                dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                 ])
           
            return childML

    return html.Div([])        
        
