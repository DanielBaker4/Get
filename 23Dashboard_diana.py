#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 13:21:46 2021

@author: diana
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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



def plot_prov(str_prov, v_prov):
        mcp = gpd.read_file('data/divisions-administratives-v1r0-comarques-1000000-20210122.shp', crs='epsg:25831')
        cols = mcp.select_dtypes(include=[np.object]).columns
        mcp[cols] = mcp[cols].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))

        client = Socrata("analisi.transparenciacatalunya.cat", None)
        results = client.get("rmgc-ncpb", limit=16774)
        # Convert to pandas DataFrame
        results_df =  gpd.GeoDataFrame.from_records(results)

        df_pobl = pd.read_excel("data/PoblacioComarques.xls")

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
        merge_bcn=merge.loc[[str_prov]]
        #eliminar duplicats
        merge_bcn = merge_bcn.drop_duplicates(subset='nomcom',keep='first')
        bcn=merge_bcn.reset_index()
        
        mcp = mcp.drop(v_prov)

        geom = mcp[["NOMCOMAR", "geometry"]] # escollir les columnes: comarca i geometria de mcp
        geom = geom.sort_values('NOMCOMAR') # ordenar alfabèticament
        Geom = gpd.GeoDataFrame(geom).set_index("NOMCOMAR") # tornem a posar l'índex
        Geom = Geom.reset_index()
        Geom = Geom.rename(columns={"NOMCOMAR": "nomcom"}) # canviar el nom

        merge2 = prov.merge(Geom, left_on='nomcom', right_on='nomcom') # associem la comarca a la província
        #filtrar Bcn
        merge2.set_index('nomdem', inplace=True)
        merge2_bcn=merge.loc[[str_prov]]
        #eliminar duplicats
        merge2_bcn = merge2_bcn.drop_duplicates(subset='nomcom',keep='first')
        bcn2=merge_bcn.reset_index()

        merge_norm = prov.merge(df_pobl, left_on='nomcom', right_on='comarca') # associem la comarca a la província
        #filtrar Bcn
        merge_norm.set_index('nomdem', inplace=True)
        merge_bcn_norm=merge_norm.loc[[str_prov]]
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
        
        return

def show_boxplot1():
    st.write('Histogram: number of accidents per hour.')
    st.write('Boxplot: average over 9 years')

    client = Socrata("analisi.transparenciacatalunya.cat", None)
    results = client.get("rmgc-ncpb", limit=16774)
    # Convert to pandas DataFrame
    df = pd.DataFrame.from_records(results)
    
    # Tenim en la columnna 'dat' el dia i l'hora juntes, volem separar-les en dues noves columnes 'data' i 'hora'
    dist_morts=df[["dat","f_morts"]]
    # Canviem la T que separa dia i hora per identificar la separació dues vegades: amb ' ' i amb '-'
    df['dat'] = df['dat'].str.replace('T',' ')
    df[['data','hora']] = df["dat"].str.split(" ", 1, expand=True)
    df['data']=df['data'].astype("datetime64")
    df["hora"] = df["hora"].str.strip(" ")
    
    df[['hores','minuts']] = df["hora"].str.split(":", 1, expand=True)
    #df["data"].dtypes

    df_hor=df[["hores"]]
    # num d'accidents en funció de la comarca i ordenats alfabèticament respecte del nom de la comarca
    accperhor = pd.value_counts(df_hor['hores'])
    accperhor = accperhor.to_frame() # convertir a Data Frame
    accperhor.index.name = 'foo'
    accperhor = accperhor.sort_values("foo") # la columna de velocitat es diu index, ordenar-la
    accperhor = accperhor.rename(columns={"hores": "# accidents"}) # canviar els noms correctes
    
    #p_bar=accperhor.plot_bokeh.bar(ylabel='# Accidents', xlabel='Hora', title="Distr d'accidents per hora",alpha=0.6)
    # Volem representar un histograma on es vegin la quantitat d'acc per hora cada any
    morts_cat=df[["any","dat","data","hores","f_morts"]]
    morts_cat.set_index("any", inplace=True)
    
    year = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
    morts_cat_any = []
    for i in range(len(year)):
        morts_cat_any.append(morts_cat.loc[[year[i]]])
    
    # num d'accidents en funció de la comarca i ordenats alfabèticament respecte del nom de la comarca
    accperhor = []
    str_acc = ["# accidents10", "# accidents11", "# accidents12", "# accidents13", "# accidents14", "# accidents15", "# accidents16", "# accidents17", "# accidents18" ]
    for i in range(len(morts_cat_any)):
        tmp = pd.value_counts(morts_cat_any[i]['hores'])
        tmp = tmp.to_frame() # convertir a Data Frame
        tmp.index.name = 'foo'
        tmp = tmp.sort_values("foo") # la columna de velocitat es diu index, ordenar-la
        tmp = tmp.rename(columns={"hores": str_acc[i]}) # canviar els noms correctes
        
        accperhor.append(tmp)
    
    accperhora=pd.concat(accperhor, axis=1)
    accperhora["mitj"]=(accperhora["# accidents10"]+accperhora["# accidents11"]+accperhora["# accidents12"]+accperhora["# accidents13"]+accperhora["# accidents14"]+accperhora["# accidents15"]+accperhora["# accidents16"]+accperhora["# accidents17"]+accperhora["# accidents18"])/9
    # accperhora["desv"]=(1/9)**(0.5)*((accperhora["# accidents10"]-accperhora["mitj"])**2+(accperhora["# accidents11"]-accperhora["mitj"])**2+(accperhora["# accidents12"]-accperhora["mitj"])**2+(accperhora["# accidents13"]-accperhora["mitj"])**2+(accperhora["# accidents14"]-accperhora["mitj"])**2+(accperhora["# accidents15"]-accperhora["mitj"])**2+(accperhora["# accidents16"]-accperhora["mitj"])**2+(accperhora["# accidents17"]-accperhora["mitj"])**2+(accperhora["# accidents18"]-accperhora["mitj"])**2)**(0.5)
    # accperhora["max"]=accperhora.max(axis=1)
    # accperhora["min"]=accperhora.min(axis=1)
    # accperhorabox=accperhora[["mitj","desv","max","min"]]
    accperhoratrans=accperhora.transpose()
    mosinteresa=accperhora[["mitj"]]
    
    #boxplot=accperhoratrans.boxplot(column=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12','13', '14', '15','16', '17', '18','19', '20', '21','22','23'])
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    lang=(mosinteresa.index)
    lang=lang.astype(int)
    lang=lang+1
    stud=mosinteresa["mitj"]
    ax.bar(lang,stud,color='pink',alpha=0.6)
    boxplot=accperhoratrans.boxplot(column=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12','13', '14', '15','16', '17', '18','19', '20', '21','22','23'], showfliers=False)
    plt.title('# accidents per hora per any')
    plt.xlabel('interval horari')
    plt.ylabel('# accidents / any')
    
    st.pyplot(fig)
    return 

def show_boxplot2():
    st.write('Histogram: number of accidents per month.')
    st.write('Boxplot: average over 9 years')

    client = Socrata("analisi.transparenciacatalunya.cat", None)
    results = client.get("rmgc-ncpb", limit=16774)
    # Convert to pandas DataFrame
    df = pd.DataFrame.from_records(results)
    
    # Tenim en la columnna 'dat' el dia i l'hora juntes, volem separar-les en dues noves columnes 'data' i 'hora'
    dist_morts=df[["dat","f_morts"]]
    # Canviem la T que separa dia i hora per identificar la separació dues vegades: amb ' ' i amb '-'
    df['dat'] = df['dat'].str.replace('T',' ')
    df[['data','hora']] = df["dat"].str.split(" ", 1, expand=True)
    df['data']=df['data'].astype("str")
    df["hora"] = df["hora"].str.strip(" ")
    
    df[['any','mesos','dies']] = df["data"].str.split("-", 2, expand=True)

    df_hor=df[["mesos"]]
    # num d'accidents en funció de la comarca i ordenats alfabèticament respecte del nom de la comarca
    accperhor = pd.value_counts(df_hor['mesos'])
    accperhor = accperhor.to_frame() # convertir a Data Frame
    accperhor.index.name = 'foo'
    accperhor = accperhor.sort_values("foo") # la columna de velocitat es diu index, ordenar-la
    accperhor = accperhor.rename(columns={"mesos": "# accidents"}) # canviar els noms correctes
        
    #p_bar=accperhor.plot_bokeh.bar(ylabel='# Accidents', xlabel='Hora', title="Distr d'accidents per hora",alpha=0.6)
    # Volem representar un histograma on es vegin la quantitat d'acc per hora cada any
    morts_cat=df[["any","dat","data","mesos","f_morts"]]
    morts_cat.set_index("any", inplace=True)
    
    year = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
    morts_cat_any = []
    for i in range(len(year)):
        morts_cat_any.append(morts_cat.loc[[year[i]]])
    
    # num d'accidents en funció de la comarca i ordenats alfabèticament respecte del nom de la comarca
    accperhor = []
    str_acc = ["# accidents10", "# accidents11", "# accidents12", "# accidents13", "# accidents14", "# accidents15", "# accidents16", "# accidents17", "# accidents18" ]
    for i in range(len(morts_cat_any)):
        tmp = pd.value_counts(morts_cat_any[i]['mesos'])
        tmp = tmp.to_frame() # convertir a Data Frame
        tmp.index.name = 'foo'
        tmp = tmp.sort_values("foo") # la columna de velocitat es diu index, ordenar-la
        tmp = tmp.rename(columns={"mesos": str_acc[i]}) # canviar els noms correctes
        
        accperhor.append(tmp)
    
    accperhora=pd.concat(accperhor, axis=1)
    accperhora["mitj"]=(accperhora["# accidents10"]+accperhora["# accidents11"]+accperhora["# accidents12"]+accperhora["# accidents13"]+accperhora["# accidents14"]+accperhora["# accidents15"]+accperhora["# accidents16"]+accperhora["# accidents17"]+accperhora["# accidents18"])/9
    # accperhora["desv"]=(1/9)**(0.5)*((accperhora["# accidents10"]-accperhora["mitj"])**2+(accperhora["# accidents11"]-accperhora["mitj"])**2+(accperhora["# accidents12"]-accperhora["mitj"])**2+(accperhora["# accidents13"]-accperhora["mitj"])**2+(accperhora["# accidents14"]-accperhora["mitj"])**2+(accperhora["# accidents15"]-accperhora["mitj"])**2+(accperhora["# accidents16"]-accperhora["mitj"])**2+(accperhora["# accidents17"]-accperhora["mitj"])**2+(accperhora["# accidents18"]-accperhora["mitj"])**2)**(0.5)
    # accperhora["max"]=accperhora.max(axis=1)
    # accperhora["min"]=accperhora.min(axis=1)
    # accperhorabox=accperhora[["mitj","desv","max","min"]]
    accperhoratrans=accperhora.transpose()
    mosinteresa=accperhora[["mitj"]]
    
    #boxplot=accperhoratrans.boxplot(column=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12','13', '14', '15','16', '17', '18','19', '20', '21','22','23'])
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    lang=(mosinteresa.index)
    lang=lang.astype(int)
    #lang=lang+1
    stud=mosinteresa["mitj"]
    ax.bar(lang,stud,color='pink',alpha=0.6)
    boxplot=accperhoratrans.boxplot(column=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], showfliers=False)
    plt.title('# accidents per mes cada any')
    plt.xlabel('mes')
    plt.ylabel('# accidents / any')
    
    st.pyplot(fig)
    
    return

def show_puntkm():
    
    st.write('Number of accidents at a given km point (here represented as the corresponding road/street percentage)')
    
    road = ['AP-7', 'N-II', 'C-31', 'N-340']
    arrange1 = [3427, 7798, 3802, 12437]
    arrange2 = [34.27, 77.98, 38.02, 12.437]
    
    for i in range(len(road)):
        st.write(road[i])
        client = Socrata("analisi.transparenciacatalunya.cat", None)
        results = client.get("rmgc-ncpb", limit=16774)
        # Convert to pandas DataFrame
        df = pd.DataFrame.from_records(results)
        # Ara agruparem per intervals de pk de carreteres
        # Fem un subdataset dels morts que n'hi ha en els accidents mortals que han ocorregut només en l'AP-7
        dist_morts_via=df[['via','pk','nomcom','f_morts']]
        
        #dist_morts_via=dist_morts_via.drop(index=dist_morts_via[dist_morts_via['f_morts']== '0'].index)
        dist_morts_via.set_index('via', inplace=True)
        dist_morts_ap=dist_morts_via.loc[[road[i]]]
        dist_morts_ap["pk"]=dist_morts_ap["pk"].astype(int)
        # Saber màxim pk
        pks=dist_morts_ap[["pk"]]
        max_val=pks.max()
        # Ordenem per punt km
        dist_morts_ap=dist_morts_ap.sort_values('pk')
        dist=dist_morts_ap.groupby(pd.cut(dist_morts_ap["pk"], np.arange(0, arrange1[i]+arrange2[i], arrange2[i]))).size()
        dist=dist.to_frame()
        dist2=dist.reset_index()
        dist2=dist2.rename(columns={"pk": "Intervals pk", 0: "#accidents"}) # canviar els noms correctes
        dist2['percentcarretera'] = dist2.index+1
        pd.options.display.float_format = "{:,.2f}".format   
    
        plt.rcParams["figure.figsize"] = 10,4
        
        x = dist2["percentcarretera"]
        y = dist2["#accidents"]
        
        fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)
        
        extent = [0,100,0,1]
        ax.imshow(y[np.newaxis,:], cmap="Greys", aspect="auto",extent=extent)
        ax.set_yticks([])
        
        ax2.plot(x,y)
        plt.title('#accidents segons el punt km de la ' + road[i])
        plt.tight_layout()
        st.pyplot(fig)
            
    return


#DASHBOARD

maps = st.container()


add_selectbox = st.sidebar.selectbox(
    "Sections",
    ("Index", "Maps","Plots", "Data, Methods and Tools","Conclusions","Author Contributions")
)


#Third:
#create page and load plots

if add_selectbox == 'Index':
    st.write('## Traffic accidents with fatalities or severely injured in Catalunya between 2010-2018')
    st.write("##### By: Diana Martínez, Daniel Panadero, Raúl Mañas")
    st.write('Traffic accidents are a big concern in all countries where the movement of vehicles is elevated. Private vehicle is by far the most unsecure way of traveling. Here we present the analysis of more than 16 thousand traffic accidents that occurred between 2010 and 2018 in Catalunya. A downward tendency is found in the fatalities […] each year. Also, the more ‘dangerous’ zone of Catalunya is found to be Lleida, and we present a hypothesis to explain that fact. […] ')

if add_selectbox == 'Author Contributions':
    st.title('Author Contributions')
    image = Image.open('Author Contributions.png')
    st.image(image)

if add_selectbox == 'Plots':
    
    st.title('Plots of the data analysis')
    
    st.sidebar.subheader("Select a plot")
    plot = st.sidebar.selectbox(
    "Plots",
    ("Box plot (acc/h/y)", "Box plot (acc/m/y)", "Punt km", "Altres")
    )
    
    
    if  plot == 'Box plot (acc/h/y)':
        show_boxplot1()
    if plot == 'Box plot (acc/m/y)':
        show_boxplot2()
    if plot == 'Punt km':
        #show_pie()
        show_puntkm()
        
    
if add_selectbox == 'Maps':
    box_prov = st.selectbox(
    "Select a Province",
    ("Barcelona","Girona","Lleida","Tarragona")
    )
    
    #llista de strings amb el nom de les provincies
    str_prov = ['Tarragona', 'Barcelona', 'Lleida', 'Girona']
    #vectors preparats per fer el drop
    v_tgn=[1,2,3,4,5,6,9,10,12,13,14,16,17,18,19,20,22,23,24,25,26,27,30,31,32,33,34,37,38,39,40,41]
    Tarragona = np.array(v_tgn)
    v_bcn=[0,1,3,4,7,8,9,11,14,15,17,18,19,21,22,24,25,26,27,28,29,30,31,32,34,35,36,37,38]
    Barcelona = np.array(v_bcn)
    v_lld=[0,1,2,5,6,7,8,9,10,11,12,15,16,18,19,20,21,23,27,28,29,30,33,35,36,39,40,41]
    Lleida = np.array(v_lld)
    v_gir=[0,2,3,4,5,6,7,8,10,11,12,13,15,16,17,20,21,22,24,25,26,28,29,31,32,34,35,36,37,38,39,40,41]
    Girona = np.array(v_gir)
    
    v_prov = [Tarragona, Barcelona, Lleida, Girona]
    
    if box_prov == str_prov[0]:
        plot_prov(str_prov[0], v_prov[0])
        
    if box_prov == str_prov[1]:
        plot_prov(str_prov[1], v_prov[1])
        
    if box_prov == str_prov[2]:
        plot_prov(str_prov[2], v_prov[2])
        
    if box_prov == str_prov[3]:
        plot_prov(str_prov[3], v_prov[3])
        
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
    
    
    
    
    