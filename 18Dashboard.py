import streamlit as st
import pandas as pd
import numpy as np
from sodapy import Socrata # Was not in the installation list for this course!
                           # Use "pip install sodapy" or "conda install sodapy" to install
import datetime
import geopandas as gpd
import json
import bokeh
from bokeh.io import show
from bokeh.models import (CDSView, ColorBar, ColumnDataSource, CustomJS,
                          CustomJSFilter, GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider)
from bokeh.layouts import column, row, widgetbox
from bokeh.palettes import brewer
from bokeh.plotting import figure
from PIL import Image



maps = st.container()


add_selectbox = st.sidebar.selectbox(
    "Sections",
    ("Index", "Maps","Data, Methods and Tools","Conclusions","Author Contributions")
)


#Third:
#create page and load plots

if add_selectbox == 'Index':
    st.write('## Traffic accidents with fatalities or severely injured in Catalunya between 2010-2018')
    st.write("##### By: Diana ??, Daniel Panadero, Raúl Mañas")
    st.write('Traffic accidents are a big concern in all countries where the movement of vehicles is elevated. Private vehicle is by far the most unsecure way of traveling. Here we present the analysis of more than 16 thousand traffic accidents that occurred between 2010 and 2018 in Catalunya. A downward tendency is found in the fatalities […] each year. Also, the more ‘dangerous’ zone of Catalunya is found to be Lleida, and we present a hypothesis to explain that fact. […] ')

if add_selectbox == 'Author Contributions':
    st.title('Author Contributions')
    image = Image.open('Author Contributions.png')
    st.image(image)

if add_selectbox == 'Maps':
    # Load file with state geometry
    contiguous_cat = gpd.read_file('data/divisions-administratives-v1r0-comarques-1000000-20210122.shp')
    # Load file with each region's population
    df_pobl = pd.read_excel("data/PoblacioComarques.xls")
    # importar el nostre dataset
    client = Socrata("analisi.transparenciacatalunya.cat", None)
    results = client.get("rmgc-ncpb", limit=17000)
    # Convert to pandas DataFrame
    results_df =  gpd.GeoDataFrame.from_records(results)
    results_df.head()

    # num d'accidents en funció de la comarca i ordenats alfabèticament respecte del nom de la comarca
    com = pd.value_counts(results_df['nomcom'])
    com = com.to_frame() # convertir a Data Frame
    com = com.reset_index() # introduir un index (0, 1, 2...)
    com = com.sort_values("index") # la columna de velocitat es diu index, ordenar-la
    com = com.rename(columns={"index": "Comarca", "nomcom": "#accidents"}) # canviar els noms correctes
    Comar = gpd.GeoDataFrame(com).set_index("Comarca")
    Comar = Comar.reset_index()
    Comar.head()

    geom = contiguous_cat[["NOMCOMAR", "geometry"]] # escollir les columnes: comarca i geometria de mcp
    geom = geom.sort_values('NOMCOMAR') # ordenar alfabèticament
    Geom = gpd.GeoDataFrame(geom).set_index("NOMCOMAR") # tornem a posar l'índex
    Geom = Geom.reset_index()
    Geom.head()

    # seleccionar columna de geometries
    geometry = Geom[['geometry']]
    # ordenem alfabèticament i deixem bonic el dataset de poblacions
    df_pobl=df_pobl.sort_values('comarca')
    df_pobl=pd.DataFrame(df_pobl).set_index('comarca')

    df_pobl=df_pobl.reset_index()
    df_pobl.head()

    # seleccionar columnes pobl i pobl_percent
    norm = df_pobl[['pobl', 'pobl_percent']]
    combo = pd.concat([Comar, geometry, norm], axis=1) # juntem comarques, #accidents, geometria i poblacions
    combo['accpercadamilhab'] = combo['#accidents']/combo['pobl']*1000 # afegim col per a #accidents per cada 1000 habitants
    combo.head()

    # Representacio interactiva
    # Input GeoJSON source that contains features for plotting
    geosource = GeoJSONDataSource(geojson = combo.to_json())

    # Define color palettes
    palette = brewer['BuGn'][8]
    palette = palette[::-1] # reverse order of colors so higher values have darker colors

    # Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
    color_mapper = LinearColorMapper(palette = palette, low = 0, high = 10)

    # Define custom tick labels for color bar.
    tick_labels = {'0': '0', '50': '500',
                   '1000':'10000', '15000':'150000',
                   '20000':'20000', '25000':'250000',
                   '30000':'30000', '350000':'350000',
                   '40000':'40000+'}

    # Create color bar.
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
                         border_line_color=None,location = (0,0), orientation = 'horizontal',
                         major_label_overrides = tick_labels)

    # Create figure object.
    p = figure(title = "Nombre d'accidents per cada 1000 habitants 2010-18", plot_height = 600 ,
               plot_width = 600, toolbar_location = 'below',
               tools = "pan, wheel_zoom, box_zoom, reset")
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # Add patch renderer to figure.
    states = p.patches('xs','ys', source = geosource,
                       fill_color = {'field' :'accpercadamilhab', 'transform' : color_mapper},
                       line_color = 'gray', line_width = 0.25, fill_alpha = 1)

    # Create hover tool
    p.add_tools(HoverTool(renderers = [states],
                          tooltips = [('Comarca','@Comarca'),('# Accidents', '@accpercadamilhab')]))

    # Specify layout
    p.add_layout(color_bar, 'below')
    p.xaxis.axis_label = 'New xlabel'
    st.bokeh_chart(p)



