#!/usr/bin/env python
# coding: utf-8

# # Visualize Geospatial Data with python
# 
# Geospatial data is simply location information that can be displayed on a map.
# There are two types of such data: Raster Data (for instance a satellite image) and Vector Data.
# Vector Data means POINTs, LINEs, POLYGONs, or collections of those (MULTIPOINT, MULTILINE, MULTIPOLYGON), but also 3D points. For instance the exact GPS coordinates of all capitals, the trajectory of a road, the boarder of a country or all borders of all districts of a city.
# Additionally, geospatial data contains attributes, e.g., the number of inhabitants, the road quality, the average income, or the mean January temperature. 
# 
# The most important python library for geospatial data in python is **geopandas**. You will see that it is structured *very* much similar to a pandas dataframe. 
# 
# Geodataframes are typically stored in shapefiles, but if you are lucky you directly get them in either .geojson or in .gpkg (GeoPackage) format, but there are of course more types. 
# Some official webpages with shapfiles:
# - countries EU: https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/countries
# - Geography Spain: https://centrodedescargas.cnig.es/CentroDescargas/index.jsp
# 
# To know how to read these files, take a look at https://geopandas.org/en/stable/docs/user_guide/io.html. Always make sure you know in which projection the geospatial data is given (Mercator, etc.). This information is encoded in the crs variable.
# 
# We already used a geopandas file when plotting covid vaccinated people per postal code.
# Let us first take load and take a look into such a geo data frame. 
# 
# # Loading geospatial Data
# As there are many different file formats for geospatial data, there are also different ways to load them.

# In[43]:



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


#First:
#create sidebar 

gender = st.sidebar.selectbox(
    "Which region do you want to display?",
    ("all", "bcn", "tgn", "gir", "lleida")
)

#scraping_depth = st.sidebar.number_input("How deep do you want to scrape?", value=2000, step=1000)


# load geojsons

# 1. geojson
# the geojson I used for the covid vaccines plot: (Most of you already downloaded ds-codigos-postales-master)
cp_barna  = gpd.read_file("data/BARCELONA.geojson")
cp_barna.head()


# The first thing we notice is that it really looks like a DataFrame. Luckily we can directly apply everything we learned about pandas on Thursday last week. The second thing we notice is the **geometry** column in this case with polygons. Every geoDataFrame has this column. The polygons in this case describe the exact boarder of each postal code area, like in the connect-the-dots images you maybe completed as a child. 

# In[44]:


# 2. geopackage 
# load some example geopackage, e.g. from https://github.com/ngageoint/GeoPackage/blob/master/docs/examples/java/example.gpkg
#example = gpd.read_file("../../example.gpkg")
#example.head()


# As you see, there can be different columns, just as in a DataFrame, but there should always be a column that carries the geometry. 

# In[45]:


# 3. shapefiles
# official shapes of Spain, e.g. downoadable from https://centrodedescargas.cnig.es/CentroDescargas/index.jsp
# For instance in the BCN500 collection, you downlowd a folder, in which you can find 
# spapes under different topics, that are documented under "Información auxiliar" in the same collection
# streets of Spain
cp_spain_mainstreets  = gpd.read_file("data/BCN500_0602L_CARRETERA_PPAL.shp", crs="epsg:4258")# streets spain, csr according to documentation of the dataset
cp_spain_smallstreets = gpd.read_file("data/BCN500_0601L_AUTOP_AUTOV.shp", crs="epsg:4258")

cp_spain_mainstreets.head() 


# In[46]:


# administrative devision
cp_spain_administr = gpd.read_file("data/BCN500_0101S_LIMITE_ADM.shp", crs="epsg:4258")# streets spain
cp_spain_administr.head()


# In[47]:


#formats Json, pckg
#read.file function of geopandas
mcp = gpd.read_file('data/divisions-administratives-v1r0-comarques-1000000-20210122.shp', crs='epsg:25831')


# In[48]:


# importar el nostre dataset
client = Socrata("analisi.transparenciacatalunya.cat", None)

results = client.get("rmgc-ncpb", limit=16774)

# Convert to pandas DataFrame
results_df =  gpd.GeoDataFrame.from_records(results)


# In[49]:


# num d'accidents en funció de la comarca
com = pd.value_counts(results_df['nomcom'])
com = com.to_frame() # convertir a Data Frame
com = com.reset_index() # introduir un index (0, 1, 2...)
com = com.sort_values("index") # la columna de velocitat es diu index, ordenar-la
com = com.rename(columns={"index": "Comarca", "nomcom": "#accidents"}) # canviar els noms correctes
Com = gpd.GeoDataFrame(com).set_index("Comarca")
Com = Com.reset_index()


# In[50]:


geom = mcp[["NOMCOMAR", "geometry"]] # escollir les columnes: comarca i geometria de mcp
geom = geom.sort_values('NOMCOMAR') # ordenar alfabèticament
Geom = gpd.GeoDataFrame(geom).set_index("NOMCOMAR") # tornem a posar l'índex
Geom = Geom.reset_index()


# In[51]:

#Third:
#create page and load plots


st.write("""# Dina
Find some interactive eeeeee on Catalunya!
""")

# seleccionar columna de geometries
geometry = Geom[['geometry']]

Junt = pd.concat([Com, geometry], axis=1) # juntem comarques, #accidents i geometria

df_poblacion = pd.read_excel("data/PoblacioComarques.xls")
# Volem ordenar alfabèticament el dataset de poblacions perquè coincideixi amb el dataset de geometries de cada comarca

df_poblacion=df_poblacion.sort_values('comarca')
df_poblacion=pd.DataFrame(df_poblacion).set_index('comarca')
df_poblacion=df_poblacion.reset_index()
# seleccionar columnes pobl i pobl_percent
norm = df_poblacion[['pobl', 'pobl_percent']]
bigJunt = pd.concat([Junt, norm], axis=1)
bigJunt['accNorm'] = bigJunt['#accidents']/bigJunt['pobl']
bigJunt['accNormpermil'] = bigJunt['accNorm']*1000
results_bigJunt =  gpd.GeoDataFrame.from_records(bigJunt)
results_bigJunt.head()
type(results_bigJunt)
results_bigJunt.crs = "epsg:25831"
results_bigJunt.plot(column='accNormpermil', legend=True,legend_kwds={'label': "Nombre d'accidents per cada mil habitants entre 2010 i 2018",
                                                              'orientation': "horizontal"})


# Instead of just staring at the dataframes, we should maybe just plot one. 
# 
# # Plotting geodataframes
# It's as simple as:

# In[52]:


cp_spain_administr.plot()


# Well, you actually only wanted Spain, but the shapefile seems to contain Algeria, Marocco, France.. as well. So let's filter, just as we already know it from pandas DataFrames.

# In[53]:


# cp_spain_administr[cp_spain_administr["TIPO_0101"]=="03"].plot()


# In[54]:


# # or you only want Catalunya
catalunya_admin = cp_spain_administr[cp_spain_administr["CCAA"]=="Cataluña"]
# catalunya_admin.plot()
# list(catalunya_admin["ID"])


# In[55]:


# # or only Barcelona
barcelona_admin = cp_spain_administr[cp_spain_administr["ETIQUETA"]=="Barcelona"]
# barcelona_admin.plot()
# list(barcelona_admin["ID"])


# In[56]:


# join 4 provincias de catalunya to catalunya
provincias_catalunya = cp_spain_administr[cp_spain_administr["CCAA"]=="Cataluña"]
streets_in_catalunya_main = cp_spain_mainstreets.sjoin(provincias_catalunya, how="inner", predicate='intersects')
streets_in_catalunya_small = cp_spain_smallstreets.sjoin(provincias_catalunya, how="inner", predicate='intersects')


# In[57]:


#This would be the easiest way but you might encounter package problems
#stations_gdf = stations_gdf.to_crs("epsg:4258")
#stations_in_barna = gpd.sjoin(stations_gdf, barcelona_admin, predicate='within')


# And of course we would like to be able to stack the different maps for streets sierras and administration in one. To this end, we need to create a matplolib canvas first, and then tell each geopandas plot function that it is ought to plot on this canvas.

# In[66]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (40, 40))
cp_spain_administr.plot(ax=ax, color = "lightgrey")
results_bigJunt.to_crs("epsg:4258").plot(ax=ax, column='accNormpermil', legend=True,legend_kwds={'label': "Nombre d'accidents per cada mil habitants entre 2010 i 2018",
                                                                                                 'orientation': "horizontal"})

# use color codes to represent the street type
streets_in_catalunya_main.plot(ax=ax, column="TIPO_0602", cmap='autumn')
streets_in_catalunya_small.plot(ax=ax, column="TIPO_0601", cmap='winter')

ax.set_xlim([-1,4])
ax.set_ylim([39.5,43.8])
ax.set_axis_off()


# In[67]:



streets_in_catalunya_main["num_TIPO_0602"] = streets_in_catalunya_main["TIPO_0602"].astype(float)
streets_in_catalunya_small["num_TIPO_0601"] = streets_in_catalunya_small["TIPO_0601"].astype(float)

figure = results_bigJunt.to_crs("epsg:4258").plot_bokeh(legend="Comarcas with population density", category='accNormpermil')
# use color codes to represent the street type
streets_in_catalunya_main.plot_bokeh(figure=figure, legend="main streets", category="num_TIPO_0602", colormap='Reds')
streets_in_catalunya_small.plot_bokeh(figure=figure, legend="minor streets", category="num_TIPO_0601", colormap='Greens')

ax.set_axis_off()


st.bokeh_chart(figure)

# In[30]:


import pandas as pd
from sodapy import Socrata

#Meteorological stations
#Metadades estacions meteorològiques automàtiques

client = Socrata("analisi.transparenciacatalunya.cat", None)
stations = client.get("yqwd-vj5e", limit=2000)

#Convert to pandas DataFrame
stations_df = pd.DataFrame.from_records(stations)
stations_gdf = gpd.GeoDataFrame(stations_df, geometry=gpd.points_from_xy(stations_df.longitud, stations_df.latitud))
stations_gdf.crs = 'epsg:4326' # the database page indicates projection "WSG84", which corresponds to crs 'epsg:4326'
stations_gdf.pop('geocoded_column')
stations_gdf.head()


# In[34]:


import matplotlib.pyplot as plt
# same code as above
#fig, ax = plt.subplots(figsize = (30, 30))
#catalunya_admin.plot(ax=ax, color = "lightgrey")
#ax.set_axis_off()

# add stations
#stations_gdf.plot(ax=ax, color = "red", markersize=10)


# In[35]:


#This would be the easiest way but you might encounter package problems
stations_gdf = stations_gdf.to_crs("epsg:4258")
stations_in_barna = gpd.sjoin(stations_gdf, barcelona_admin, predicate='within')

import matplotlib.pyplot as plt
# same code as above
fig, ax = plt.subplots(figsize = (10, 10))
ax.set_axis_off()
barcelona_admin.plot(ax=ax, color = "aliceblue")
# add stations
stations_in_barna.plot(ax=ax, color = "red", markersize=7)


# We saw the very basic of how to load, manipulate and plot geospatial data in python. As you saw in former sessions, geopandas DataFrames can also be plotted with bokeh in one line.

# In[36]:


import pandas_bokeh
stations_gdf.plot_bokeh()


# An other straightforward way is by using contextily, which you need to install first. pip install contextily.
# 
# Docs: https://contextily.readthedocs.io/en/latest/intro_guide.html

# In[37]:


import contextily
fig, ax = plt.subplots(figsize = (15, 20))
stations_in_barna.plot(ax=ax, color = "red", marker="$St.$", markersize=90)
contextily.add_basemap(ax, crs=stations_in_barna.crs.to_string())
ax.set_axis_off()


# In[ ]:




