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
from PIL import Image

# Load data
cp_spain_administr = gpd.read_file("data/BCN500_0101S_LIMITE_ADM.shp", crs="epsg:4258")# streets spain
cp_spain_mainstreets  = gpd.read_file("data/BCN500_0602L_CARRETERA_PPAL.shp", crs="epsg:4258")# streets spain, csr 
#according to documentation of the dataset
cp_spain_smallstreets = gpd.read_file("data/BCN500_0601L_AUTOP_AUTOV.shp", crs="epsg:4258")
provincias_catalunya = cp_spain_administr[cp_spain_administr["CCAA"]=="Cataluña"]
streets_in_catalunya_main =  gpd.read_file("data/streets_in_catalunya_main.shp", crs="epsg:4258")
streets_in_catalunya_small = gpd.read_file("data/streets_in_catalunya_small.shp", crs="epsg:4258")
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
df =  pd.DataFrame.from_records(results)

df_pobl = pd.read_excel("data/PoblacioComarques.xls")

#Define functions for showing plots
def show_boxplot1():
    st.write('Histogram: number of accidents per hour.')
    st.write('Boxplot: average over 9 years')

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

#Dashboard page

add_selectbox = st.sidebar.selectbox(
    "Sections",
    ("Index", "Maps","Plots","Data, Methods and Tools","Conclusions","Author Contributions")
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
    
    #create display bar 
    
    region = ("All", "Barcelona", "Tarragona", "Girona", "Lleida")
    options = list(range(len(region)))
    
    value = st.selectbox("Which region do you want to display?", options, format_func=lambda x: region[x])
    
        # gender translation
    dict_value = {0: "All",
                   1: "Barcelona",
                   2: "Tarragona", 
                   3: "Girona", 
                   4: "Lleida"}
    
    # filter plot for region
    if dict_value[value] in ["Barcelona", "Tarragona", "Girona", "Lleida"]:
        # Ara fem la sel·lecció de provincies a representar en el mapa Bokeh
        
        com = pd.value_counts(df['nomcom'])
        com = com.to_frame() # convertir a Data Frame
        com = com.reset_index() # introduir un index (0, 1, 2...)
        com = com.sort_values("index") # la columna de velocitat es diu index, ordenar-la
        com = com.rename(columns={"index": "nomcom", "nomcom": "#accidents"}) # canviar els noms correctes
        Comar = gpd.GeoDataFrame(com).set_index("nomcom")
        Comar = Comar.reset_index()
        
        prov = df[['nomcom', 'nomdem']]
        prov = prov.sort_values('nomcom') 
        
        merge = prov.merge(Comar, left_on='nomcom', right_on='nomcom') # associem la comarca a la província, merge 16 mil
        #filtrar
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
        # Geom = Geom.rename(columns={"NOMCOMAR": "nomcom"}) # canviar el nom
        
        merge2 = prov.merge(Geom, left_on='nomcom', right_on='NOMCOMAR') # associem la comarca a la província
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
        
        
        temps_com = df[['nomdem','nomcom','d_superficie']]
        temps_com = temps_com.groupby(['nomcom', 'd_superficie']).size()
        temps_com = temps_com.to_frame()
        temps_com =temps_com.unstack()
        temps_com = temps_com.fillna(0)
        temps_com = temps_com.reset_index()
        temps_com.columns = temps_com.columns.get_level_values(1)
        temps_com['percent_mal_estat'] = ((temps_com['Gelat']+temps_com['Inundat']+
                                           temps_com['Mullat']+temps_com['Nevat']+temps_com['Relliscós'])/(temps_com['Sec i net']+temps_com['Gelat']+temps_com['Inundat']+
                                           temps_com['Mullat']+temps_com['Nevat']+temps_com['Relliscós']))*100
        temps_com=temps_com.rename(columns={'':'nomcom'})
        merge_temps = prov.merge(temps_com, left_on='nomcom', right_on='nomcom') # associem la comarca a la província
        # filtrar Bcn
        merge_temps.set_index('nomdem', inplace=True)
        merge_temps_bcn=merge_temps.loc[[dict_value[value]]]
        #eliminar duplicats
        merge_temps_bcn = merge_temps_bcn.drop_duplicates(subset='nomcom',keep='first')
        bcn_temps = merge_temps_bcn.reset_index()
                
        temps = bcn_temps[['nomcom', 'percent_mal_estat']]
        
        hora_com = df[['nomdem','nomcom','gruphor']]
        hora_com = hora_com.groupby(['nomcom', 'gruphor']).size()
        hora_com = hora_com.to_frame()
        hora_com = hora_com.unstack()
        hora_com.columns = hora_com.columns.get_level_values(1)
        hora_com['percent_nit'] = ((hora_com['Nit'])/(hora_com['Matí']+hora_com['Nit']+hora_com['Tarda']))*100
        hora_com = hora_com.reset_index(level = 0)
        
        merge_hora = prov.merge(hora_com, left_on='nomcom', right_on='nomcom') # associem la comarca a la província
        # filtrar Bcn
        merge_hora.set_index('nomdem', inplace=True)
        merge_hora_bcn=merge_hora.loc[[dict_value[value]]]
        #eliminar duplicats
        merge_hora_bcn = merge_hora_bcn.drop_duplicates(subset='nomcom',keep='first')
        bcn_hora = merge_hora_bcn.reset_index()
                
        hora = bcn_hora[['nomcom', 'percent_nit']]
        
        combo = pd.concat([bcn, Geom, norm, temps, hora], axis=1) # juntem comarques, #accidents, geometria, poblacions i percentatges d'accidents amb calçada millorable i nocturns de la provincia en qüestió
        combo['accpercadamilhab'] = combo['#accidents']/combo['pobl']*1000 # afegim col per a #accidents per cada 1000 habitants
        # combo = combo[['#accidents', 'nomcom', 'geometry', 'accpercadamilhab']]
        combo = combo.rename(columns={"NOMCOMAR": "Comarca", "accpercadamilhab": "# Acc. Norm."}) # canviar els noms correctes
        combo=combo.dropna()
        # combo
        
        combo =  gpd.GeoDataFrame.from_records(combo)
        combo.crs = "epsg:25831"
        p = combo.to_crs("epsg:4258").plot_bokeh(legend=False,title="Comarcas with car accident density",category='# Acc. Norm.',hovertool_columns=['Comarca','# Acc. Norm.'],show_figure=False)
        
        gdf_prov.plot_bokeh(figure=p,hovertool_string="""@City""",marker="inverted_triangle",size=10,color="yellow",show_figure=False)
        # streets_in_catalunya_main.plot_bokeh(figure=p, legend="main streets", category="num_TIPO_0602", colormap='Reds',show_figure=False)
        # streets_in_catalunya_small.plot_bokeh(figure=p, legend="minor streets", category="num_TIPO_0601", colormap='Greens',show_figure=False)
        
        st.bokeh_chart(p)
    else:
        # num d'accidents en funció de la comarca
        com = pd.value_counts(df['nomcom'])
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
        temps_com = df[['nomcom','d_superficie']]
        temps_com = temps_com.groupby(['nomcom', 'd_superficie']).size()
        temps_com = temps_com.to_frame()
        temps_com =temps_com.unstack()
        temps_com = temps_com.fillna(0)
        temps_com = temps_com.reset_index()
        temps_com.columns = temps_com.columns.get_level_values(1)
        temps_com['percent_mal_estat'] = ((temps_com['Gelat']+temps_com['Inundat']+
                                   temps_com['Mullat']+temps_com['Nevat']+temps_com['Relliscós'])/(temps_com['Sec i net']+temps_com['Gelat']+temps_com['Inundat']+
                                   temps_com['Mullat']+temps_com['Nevat']+temps_com['Relliscós']))*100
        combo['percent_mal_estat'] = temps_com['percent_mal_estat']                                                                                           
        hora_com = df[['nomcom','gruphor']]
        hora_com = hora_com.groupby(['nomcom', 'gruphor']).size()
        hora_com = hora_com.to_frame()
        hora_com = hora_com.unstack()
        hora_com.columns = hora_com.columns.get_level_values(1)
        hora_com['percent_nit'] = ((hora_com['Nit'])/(hora_com['Matí']+hora_com['Nit']+hora_com['Tarda']))*100
        hora_com
        hora_com = hora_com.reset_index(level = 0)
        combo['percent_nit'] = hora_com['percent_nit']
        combo = pd.concat([Com,geometry,norm], axis=1)
        combo['AccNorm'] = (combo['#accidents']/combo['pobl'])*1000
        combo = combo.rename(columns={"nomcom": "Comarca", "AccNorm": "# Acc. Norm."}) # canviar els noms correctes
        combo =  gpd.GeoDataFrame.from_records(combo)
        combo.crs = "epsg:25831"
        p = combo.to_crs("epsg:4258").plot_bokeh(legend=False,title="Comarcas with car accident density", category='# Acc. Norm.',hovertool_columns=['Comarca','# Acc. Norm.'],show_figure=False)
     
        gdf_prov.plot_bokeh(figure=p,hovertool_string="""@City""",marker="inverted_triangle",size=10,color="yellow",show_figure=False)
       
        st.bokeh_chart(p)



