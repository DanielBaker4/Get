# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 23:59:13 2021

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
#create sidebar 

region = st.sidebar.selectbox(
    "Which region do you want to display?",
    ("All", "Barcelona", "Tarragona", "Girona", "Lleida")
)

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
# Meteorological stations
client = Socrata("analisi.transparenciacatalunya.cat", None)
stations = client.get("yqwd-vj5e", limit=2000)
# Convert to pandas DataFrame
stations_df = pd.DataFrame.from_records(stations)
stations_gdf = gpd.GeoDataFrame(stations_df, geometry=gpd.points_from_xy(stations_df.longitud, stations_df.latitud))
stations_gdf.crs = 'epsg:4326' # the database page indicates projection "WSG84", which corresponds to crs 'epsg:4326'
stations_gdf = stations_gdf.to_crs("epsg:4258")
df_cities = pd.DataFrame({'City': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'],'Country': ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Venezuela'],'Latitude': [-34.58, -15.78, -33.45, 4.60, 10.48],'Longitude': [-58.66, -47.91, -70.66, -74.08, -66.86]})
gdf_cities = gpd.GeoDataFrame(df_cities, geometry=gpd.points_from_xy(df_cities.Longitude, df_cities.Latitude))
gdf_cities.crs='epsg:4258'
gdf_cities = gdf_cities.to_crs("epsg:4258")
df_prov = pd.DataFrame({'City': ['Barcelona', 'Tarragona', 'Girona', 'Lleida'],'Latitude': [41.73, 41.08, 42.128, 42.04],'Longitude': [1.984, 0.818, 2.6735, 1.0479]})
gdf_prov = gpd.GeoDataFrame(df_prov, geometry=gpd.points_from_xy(df_prov.Longitude, df_prov.Latitude))
gdf_prov.crs='epsg:4258'
gdf_prov = gdf_prov.to_crs("epsg:4258")

mcp = gpd.read_file('data/divisions-administratives-v1r0-comarques-1000000-20210122.shp', crs='epsg:25831')
client = Socrata("analisi.transparenciacatalunya.cat", None)
results = client.get("rmgc-ncpb", limit=16774)
# Convert to pandas DataFrame
results_df =  gpd.GeoDataFrame.from_records(results)

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
df_poblacion = pd.read_excel("data/PoblacioComarques.xls")
# Volem ordenar alfabèticament el dataset de poblacions perquè coincideixi amb el dataset de geometries de cada comarca
df_poblacion=df_poblacion.sort_values('comarca')
df_poblacion=pd.DataFrame(df_poblacion).set_index('comarca')
df_poblacion=df_poblacion.reset_index()
# seleccionar columnes pobl i pobl_percent
norm = df_poblacion[['pobl', 'pobl_percent']]
combo = pd.concat([Com,geometry,norm], axis=1)
combo['AccNorm'] = (combo['#accidents']/combo['pobl'])*1000
combo =  gpd.GeoDataFrame.from_records(combo)
combo.crs = "epsg:25831"

# Apply filters

# Convert to pandas DataFrame

# gender translation
#dict_region = {"bcn":"Barcelona", "tgn":"Tarragona", "gir":"Girona", "lle":"Lleida",
#               "all": "All"}
# filter dataframe for gender
#if dict_region[region] in ["Barcelona", "Tarragona", "Girona", "Lleida"]:
#    filtered_df = vacc_ppc_df[vacc_ppc_df["region"]==dict_region[region]]
#else:
#    filtered_df = vacc_ppc_df

#Third:
#create page and load plots


st.write("""# Group 3: Traffic accidents with deaths or injuries
Raul M., Diana M. & Daniel P.
""")

# Bokeh plot with vaccinated per postal_code
#pc = list(filtered_df["municipi_codi"])

#p = plot_abundance_for_list_of_postal_codes(pc)


p = combo.to_crs("epsg:4258").plot_bokeh(legend="Comarcas with car accident density", category='AccNorm',show_figure=False)
# use color codes to represent the street type
# stations_gdf.plot_bokeh(figure=p, legend="Meteo stations",color="red",show_figure=False)
# gdf_cities.plot_bokeh(figure=p,hovertool_string="""@City @Country""",marker="inverted_triangle",color="red",show_figure=False)
gdf_prov.plot_bokeh(figure=p,hovertool_string="""@City""",marker="inverted_triangle",size=10,color="yellow",show_figure=False)
# streets_in_catalunya_main.plot_bokeh(figure=p, legend="main streets", category="num_TIPO_0602", colormap='Reds',show_figure=False)
# streets_in_catalunya_small.plot_bokeh(figure=p, legend="minor streets", category="num_TIPO_0601", colormap='Greens',show_figure=False)

st.bokeh_chart(p)