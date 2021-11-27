# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:07:45 2021

@author: Daniel
"""

import streamlit as st

import os
import geopandas as gpd
from pprint import pprint
import numpy as np
import pandas as pd
import sodapy as sp
import pandas_bokeh as pbk
pbk.output_notebook()
from sodapy import Socrata
import matplotlib.pyplot as plt
import geopandas as gpd
import descartes 
from bokeh.plotting import figure, output_file, show

#First:

#Third:
#create page and load plots


st.write("""# Group 3: Traffic accidents with deaths or injuries
Raul M., Diana M. & Daniel P.
""")

#create display bar 

region = ("All", "Barcelona", "Tarragona", "Girona", "Lleida")
options = list(range(len(region)))

value = st.selectbox("Which region do you want to display?", options, format_func=lambda x: region[x])

# Second:
# Load data and apply filters according to settings in sidebar

# Load data
cp_spain_administr = gpd.read_file("data/BCN500_0101S_LIMITE_ADM.shp", crs="epsg:4258")# streets spain
cp_spain_mainstreets  = gpd.read_file("data/BCN500_0602L_CARRETERA_PPAL.shp", crs="epsg:4258")# streets spain, csr 
#according to documentation of the dataset
cp_spain_smallstreets = gpd.read_file("data/BCN500_0601L_AUTOP_AUTOV.shp", crs="epsg:4258")
provincias_catalunya = cp_spain_administr[cp_spain_administr["CCAA"]=="Cataluña"]
streets_in_catalunya_main =  gpd.read_file("streets_in_catalunya_main.shp", crs="epsg:4258")
streets_in_catalunya_small = gpd.read_file("streets_in_catalunya_small.shp", crs="epsg:4258")
streets_in_catalunya_main["num_TIPO_0602"] = streets_in_catalunya_main["TIPO_0602"].astype(float)
streets_in_catalunya_small["num_TIPO_0601"] = streets_in_catalunya_small["TIPO_0601"].astype(float)

df_prov = pd.DataFrame({'City': ['Barcelona', 'Tarragona', 'Girona', 'Lleida'],'Latitude': [41.73, 41.08, 42.128, 42.04],'Longitude': [1.984, 0.818, 2.6735, 1.0479]})
gdf_prov = gpd.GeoDataFrame(df_prov, geometry=gpd.points_from_xy(df_prov.Longitude, df_prov.Latitude))
gdf_prov.crs='epsg:4258'
gdf_prov = gdf_prov.to_crs("epsg:4258")

mcp = gpd.read_file('data/divisions-administratives-v1r0-comarques-1000000-20210122.shp', crs='epsg:25831')
client = Socrata("analisi.transparenciacatalunya.cat", None)
results = client.get("rmgc-ncpb", limit=16774)
# Convert to pandas DataFrame
results_df =  gpd.GeoDataFrame.from_records(results)

df_pobl = pd.read_excel("PoblacioComarques.xls")

# gender translation
dict_value = {0: "All",
               1: "Barcelona",
               2: "Tarragona", 
               3: "Girona", 
               4: "Lleida"}

# filter plot for region
if dict_value[value] in ["Barcelona", "Tarragona", "Girona", "Lleida"]:
    # Ara fem la sel·lecció de provincies a representar en el mapa Bokeh
    
    com = pd.value_counts(results_df['nomcom'])
    com = com.to_frame() # convertir a Data Frame
    com = com.reset_index() # introduir un index (0, 1, 2...)
    com = com.sort_values("index") # la columna de velocitat es diu index, ordenar-la
    com = com.rename(columns={"index": "nomcom", "nomcom": "#accidents"}) # canviar els noms correctes
    Comar = gpd.GeoDataFrame(com).set_index("nomcom")
    Comar = Comar.reset_index()
    
    prov = results_df[['nomcom', 'nomdem']]
    prov = prov.sort_values('nomcom') 
    
    merge = prov.merge(Comar, left_on='nomcom', right_on='nomcom') # associem la comarca a la província, merge 16 mil
    #filtrar Bcn
    merge.set_index('nomdem', inplace=True)
    merge_bcn=merge.loc[[dict_value[value]]]
    #eliminar duplicats
    merge_bcn = merge_bcn.drop_duplicates(subset='nomcom',keep='first')
    bcn=merge_bcn.reset_index()
    
    # st.write(dict_value[value])
    # st.write(type(dict_value[value]))
    if dict_value[value] in ["Barcelona", "Tarragona", "Girona"]:
        if dict_value[value]  in ["Barcelona", "Tarragona"]:
            if dict_value[value] == "Barcelona":
                v1=[0,1,3,4,7,8,9,11,14,15,17,18,19,21,22,24,25,26,27,28,29,30,31,32,34,35,36,37,38]            
                v = np.array(v1)
            else:
                v2=[1,2,3,4,5,6,9,10,12,13,14,16,17,18,19,20,22,23,24,25,26,27,30,31,32,33,34,37,38,39,40,41]
                v= np.array(v2)
        else:
            v3=[0,2,3,4,5,6,7,8,10,11,12,13,15,16,17,20,21,22,24,25,26,28,29,31,32,34,35,36,37,38,39,40,41]
            v = np.array(v3)
    else:
        v4=[0,1,2,5,6,7,8,9,10,11,12,15,16,18,19,20,21,23,27,28,29,30,33,35,36,39,40,41]
        v = np.array(v4)
            
    mcp = mcp.drop(v)
    
    geom = mcp[["NOMCOMAR", "geometry"]] # escollir les columnes: comarca i geometria de mcp
    geom = geom.sort_values('NOMCOMAR') # ordenar alfabèticament
    Geom = gpd.GeoDataFrame(geom).set_index("NOMCOMAR") # tornem a posar l'índex
    Geom = Geom.reset_index()
    Geom = Geom.rename(columns={"NOMCOMAR": "nomcom"}) # canviar el nom
    
    merge2 = prov.merge(Geom, left_on='nomcom', right_on='nomcom') # associem la comarca a la província
    #filtrar Bcn
    merge2.set_index('nomdem', inplace=True)
    merge2_bcn=merge.loc[[dict_value[value]]]
    #eliminar duplicats
    merge2_bcn = merge2_bcn.drop_duplicates(subset='nomcom',keep='first')
    bcn2=merge_bcn.reset_index()
    
    merge_norm = prov.merge(df_pobl, left_on='nomcom', right_on='comarca') # associem la comarca a la província
    #filtrar Bcn
    merge_norm.set_index('nomdem', inplace=True)
    merge_bcn_norm=merge_norm.loc[[dict_value[value]]]
    #eliminar duplicats
    merge_bcn_norm = merge_bcn_norm.drop_duplicates(subset='nomcom',keep='first')
    bcn_norm = merge_bcn_norm.reset_index()
    
    norm = bcn_norm[['pobl', 'pobl_percent']]
    combo = pd.concat([bcn, Geom, norm], axis=1) # juntem comarques, #accidents, geometria i poblacions de bcn
    combo['accpercadamilhab'] = combo['#accidents']/combo['pobl']*1000 # afegim col per a #accidents per cada 1000 habitants
    # combo = combo[['#accidents', 'nomcom', 'geometry', 'accpercadamilhab']]
    combo=combo.dropna()
    # combo
    
    combo =  gpd.GeoDataFrame.from_records(combo)
    combo.crs = "epsg:25831"
    p = combo.to_crs("epsg:4258").plot_bokeh(legend="Comarcas with car accident density", category='accpercadamilhab',show_figure=False)
    
    gdf_prov.plot_bokeh(figure=p,hovertool_string="""@City""",marker="inverted_triangle",size=10,color="yellow",show_figure=False)
    # streets_in_catalunya_main.plot_bokeh(figure=p, legend="main streets", category="num_TIPO_0602", colormap='Reds',show_figure=False)
    # streets_in_catalunya_small.plot_bokeh(figure=p, legend="minor streets", category="num_TIPO_0601", colormap='Greens',show_figure=False)
    
    st.bokeh_chart(p)
else:
    # num d'accidents en funció de la comarca
    com = pd.value_counts(results_df['nomcom'])
    com = com.to_frame() # convertir a Data Frame
    com = com.reset_index() # introduir un index (0, 1, 2...)
    com = com.sort_values("index") # la columna de velocitat es diu index, ordenar-la
    com = com.rename(columns={"index": "Comarca", "nomcom": "#accidents"}) # canviar els noms correctes
    Com = gpd.GeoDataFrame(com).set_index("Comarca")
    Com = Com.reset_index()
    geom = mcp[["NOMCOMAR", "geometry"]] # escollir les columnes: comarca i geometria de mcp
    geom = geom.sort_values('NOMCOMAR') # ordenar alfabèticament
    Geom = gpd.GeoDataFrame(geom).set_index("NOMCOMAR") # tornem a posar l'índex
    Geom = Geom.reset_index()
    # seleccionar columna de geometries
    geometry = Geom[['geometry']]
    # Volem ordenar alfabèticament el dataset de poblacions perquè coincideixi amb el dataset de geometries de cada comarca
    df_poblacion=df_pobl.sort_values('comarca')
    df_poblacion=pd.DataFrame(df_poblacion).set_index('comarca')
    df_poblacion=df_poblacion.reset_index()
    # seleccionar columnes pobl i pobl_percent
    norm = df_poblacion[['pobl', 'pobl_percent']]
    combo = pd.concat([Com,geometry,norm], axis=1)
    combo['AccNorm'] = (combo['#accidents']/combo['pobl'])*1000
    combo =  gpd.GeoDataFrame.from_records(combo)
    combo.crs = "epsg:25831"
    p = combo.to_crs("epsg:4258").plot_bokeh(legend="Comarcas with car accident density", category='AccNorm',show_figure=False)
 
    gdf_prov.plot_bokeh(figure=p,hovertool_string="""@City""",marker="inverted_triangle",size=10,color="yellow",show_figure=False)
   
    st.bokeh_chart(p)
