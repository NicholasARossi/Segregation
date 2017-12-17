import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


mpl.rcParams['pdf.fonttype']=42
sns.set_style('white')

### gradient functions
def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])
def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return RGB_list
num_cells=10







if __name__== "__main__" :
    #import school data
    data = pd.read_csv("data2.csv", encoding="ISO-8859-1")

    #parse specific states
    CA_data = data.where(data['LEA_STATE'] == 'CA')
    CA_data = CA_data.dropna()
    NY_data = data.where(data['LEA_STATE'] == 'NY')
    NY_data = NY_data.dropna()
    NJ_data = data.where(data['LEA_STATE'] == 'NJ')
    NJ_data = NJ_data.dropna()
    MA_data = data.where(data['LEA_STATE'] == 'MA')
    MA_data = MA_data.dropna()

    ### California Analysis


    ### Cali Map : california percentage white is 39%

    plt.close('all')
    fig, ax1 = plt.subplots(figsize=(12, 12))


    colors = linear_gradient("#FF00FF", "#808080", n=39)
    colors2 = linear_gradient("#808080", "#00CED1", n=61)
    color_list = colors + colors2


    map = Basemap(resolution='l', llcrnrlon=-123, llcrnrlat=30, urcrnrlon=-113, urcrnrlat=43,
                  projection='lcc', lat_1=80, lat_2=80, lon_0=-115, ax=ax1)

    map.readshapefile('st99_d00', name='states', linewidth=5, color='slategrey', antialiased=1, ax=ax1, drawbounds=True)
    map.drawmapboundary(fill_color='#DCDCDC')
    map.fillcontinents(color='white')


    colors = ['darkturquoise', 'magenta', 'grey']

    for r in range(len(CA_data)):

        x, y = map(CA_data['CCD_LONCOD'].iloc[r], CA_data['CCD_LATCOD'].iloc[r])
        if CA_data['SCH_ENR_WH_M'].iloc[r]:
            map.plot(x,y,marker='o',color=np.divide(color_list[int(CA_data['SCH_ENR_WH_M'].iloc[r]/CA_data['TOT_ENR_M'].iloc[r]*100)-1],255),markersize=20*CA_data['TOT_ENR_M'].iloc[r]/2000,alpha=0.25)

    for l in np.arange(5) + 1:
        x, y = map(min(CA_data['CCD_LONCOD']) + 1 * l + 1, min(CA_data['CCD_LATCOD']) - 1)
        map.plot(x, y, marker='o', color=colors[l % 3], markersize=10 * l)

    fig.savefig('figures/caltemp.pdf')

    ### Cali Analytics
    plt.close('all')

    fig, ax = plt.subplots()
    x = CA_data['SCH_ENR_WH_M'] + CA_data['SCH_ENR_WH_F']
    tot = (CA_data['TOT_ENR_M'] + CA_data['TOT_ENR_F'])
    y = tot - x

    colors = linear_gradient("#FF00FF", "#808080", n=39)
    colors2 = linear_gradient("#808080", "#00CED1", n=62)
    color_list2 = colors + colors2
    listo = []
    sizes = []
    for r in range(len(y)):
        if tot.iloc[r]:
            listo.append(np.divide(color_list2[int(x.iloc[r] / tot.iloc[r] * 100)], 255))
            sizes.append(100 * tot.iloc[r] / 1000)
        else:
            listo.append(np.divide(color_list2[0], 255))
            sizes.append(0)
    ax.scatter(x, y, alpha=0.15, color=listo, s=sizes)
    x_vect = np.linspace(0, 1000, 1000)
    array = np.linspace(1, 5000, 5000)
    ax.plot(array * .39, array * .61, color='black', linewidth=5)

    ax.set_xlim([-100, 1000])
    ax.set_ylim([-100, 1500])
    ax.set_xlabel('Number of White Students')
    ax.set_ylabel('Number of Non-White Students')

    plt.tight_layout()
    plt.axis('off')
    fig.savefig('figures/cali_analytics.png')


    ### NY Map

    plt.close('all')
    fig, ax1 = plt.subplots(figsize=(12, 12))

    colors = linear_gradient("#FF00FF", "#808080", n=57)
    colors2 = linear_gradient("#808080", "#00CED1", n=43)
    color_list = colors + colors2

    map = Basemap(resolution='l', llcrnrlon=-80, llcrnrlat=40.5, urcrnrlon=-70, urcrnrlat=45,
                  projection='lcc', lat_1=80, lat_2=80, lon_0=-80, ax=ax1)

    map.readshapefile('st99_d00', name='states', linewidth=5, color='slategrey', antialiased=1, ax=ax1, drawbounds=True)
    map.drawmapboundary(fill_color='#DCDCDC')
    map.fillcontinents(color='white')

    colors = ['darkturquoise', 'magenta', 'grey']
    for r in range(len(NY_data)):

        x, y = map(NY_data['CCD_LONCOD'].iloc[r], NY_data['CCD_LATCOD'].iloc[r])
        if NY_data['SCH_ENR_WH_M'].iloc[r]:
            map.plot(x, y, marker='o', color=np.divide(
                color_list[int(NY_data['SCH_ENR_WH_M'].iloc[r] / NY_data['TOT_ENR_M'].iloc[r] * 100) - 1], 255),
                     markersize=30 * NY_data['TOT_ENR_M'].iloc[r] / 2000, alpha=0.25)

    for l in np.arange(3) + 1:
        x, y = map(min(NY_data['CCD_LONCOD']) + l * .75, min(NY_data['CCD_LATCOD']) + 1)
        map.plot(x + 40, y, marker='o', color=colors[l % 3], markersize=15 * l)
    fig.savefig('figures/nyfig.pdf')

    ### NY analytics:
    plt.close('all')

    fig, ax = plt.subplots()
    x = NY_data['SCH_ENR_WH_M'] + NY_data['SCH_ENR_WH_F']
    tot = (NY_data['TOT_ENR_M'] + NY_data['TOT_ENR_F'])
    y = tot - x

    colors = linear_gradient("#FF00FF", "#808080", n=57)
    colors2 = linear_gradient("#808080", "#00CED1", n=44)
    color_list2 = colors + colors2
    listo = []
    sizes = []
    for r in range(len(y)):
        if tot.iloc[r]:
            listo.append(np.divide(color_list2[int(x.iloc[r] / tot.iloc[r] * 100)], 255))
            sizes.append(100 * tot.iloc[r] / 1000)
        else:
            listo.append(np.divide(color_list2[0], 255))
            sizes.append(0)
    ax.scatter(x, y, alpha=0.15, color=listo, s=sizes)
    x_vect = np.linspace(0, 2000, 1000)

    array = np.linspace(1, 5000, 5000)
    ax.plot(array * .57, array * .44, color='black', linewidth=5)

    ax.set_xlim([-100, 1000])
    ax.set_ylim([-100, 1500])
    ax.set_xlabel('Number of White Students')
    ax.set_ylabel('Number of Non-White Students')

    plt.tight_layout()
    plt.axis('off')

    fig.savefig('figures/newyork_analytics.png', dpi=300)

    ### NYC map:
    plt.close('all')

    import matplotlib.pyplot as plt
    import shapefile

    # from helpers import slug
    from mpl_toolkits.basemap import Basemap

    shpfile = 'shapefiles/new-york_new-york_osm_roads'
    fontcolor = '#666666'

    fig = plt.figure(figsize=(28, 20))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)

    sf = shapefile.Reader(shpfile)

    x0, y0, x1, y1 = sf.bbox

    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    m = Basemap(llcrnrlon=x0, llcrnrlat=y0, urcrnrlon=x1, urcrnrlat=y1, lat_0=cx, lon_0=cy, resolution='c',
                projection='mill')
    m.drawmapboundary(fill_color='#ffffff', linewidth=.0)

    m.readshapefile(shpfile, 'metro', linewidth=.15)

    for r in range(len(NY_data)):

        x, y = m(NY_data['CCD_LONCOD'].iloc[r], NY_data['CCD_LATCOD'].iloc[r])
        if NY_data['SCH_ENR_WH_M'].iloc[r]:
            m.plot(x, y, marker='o', color=np.divide(
                color_list[int(NY_data['SCH_ENR_WH_M'].iloc[r] / NY_data['TOT_ENR_M'].iloc[r] * 100) - 1], 255),
                   markersize=30 * NY_data['TOT_ENR_M'].iloc[r] / 2000, alpha=0.75)

    for r in range(len(NJ_data)):

        x, y = m(NJ_data['CCD_LONCOD'].iloc[r], NJ_data['CCD_LATCOD'].iloc[r])
        if NJ_data['SCH_ENR_WH_M'].iloc[r]:
            m.plot(x, y, marker='o', color=np.divide(
                color_list[int(NJ_data['SCH_ENR_WH_M'].iloc[r] / NJ_data['TOT_ENR_M'].iloc[r] * 100) - 1], 255),
                   markersize=30 * NJ_data['TOT_ENR_M'].iloc[r] / 2000, alpha=0.75)

    fig.savefig('figures/nycfig.pdf', dpi=200)


    ### data for interactives
    temp_array = np.zeros((1, 51))

    d = {'state': ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
                   "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
                   "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
        , 'per_white': np.array(
            [0.67, 0.63, 0.57, 0.74, 0.39, 0.69, 0.70, 0.64, 0.35, 0.57, 0.55, 0.23, 0.83, 0.63, 0.81, 0.88, 0.77, 0.86,
             0.60, 0.94, 0.54, 0.75, 0.79, 0.82, 0.58, 0.81, 0.87, 0.81, 0.53, 0.92, 0.58, 0.40, 0.57, 0.65, 0.88, 0.81,
             0.68, 0.78, 0.79, 0.75, 0.64, 0.84, 0.75, 0.44, 0.80, 0.94, 0.64, 0.71, 0.93, 0.83, 0.85]),
         'seg': temp_array[0]}
    whitenas = pd.DataFrame(d)

    states = data.LEA_STATE.unique()
    errors = []
    averages = []
    temp_array = np.zeros((1, 51))
    for state in states:

        temp_dat = data[data['LEA_STATE'] == state]
        true = float(whitenas[whitenas['state'] == state].per_white)
        error = 0
        for r in range(len(temp_dat)):
            white = temp_dat['SCH_ENR_WH_M'].iloc[r] + temp_dat['SCH_ENR_WH_F'].iloc[r]
            total = temp_dat['TOT_ENR_M'].iloc[r] + temp_dat['TOT_ENR_F'].iloc[r]
            observed = white / total
            error += np.nan_to_num((abs(observed - true) / np.sqrt((1 / total) * (true * (1 - true)))))
        errors.append(error / len(temp_dat))
        averages.append(true)
        temp_array[0, np.where(whitenas['state'] == state)[0][0]] = error / len(temp_dat)


    d = {'state': ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
                   "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
                   "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
        , 'per_white': np.array(
            [0.67, 0.63, 0.57, 0.74, 0.39, 0.69, 0.70, 0.64, 0.35, 0.57, 0.55, 0.23, 0.83, 0.63, 0.81, 0.88, 0.77, 0.86,
             0.60, 0.94, 0.54, 0.75, 0.79, 0.82, 0.58, 0.81, 0.87, 0.81, 0.53, 0.92, 0.58, 0.40, 0.57, 0.65, 0.88, 0.81,
             0.68, 0.78, 0.79, 0.75, 0.64, 0.84, 0.75, 0.44, 0.80, 0.94, 0.64, 0.71, 0.93, 0.83, 0.85]),
         'seg': temp_array[0]}

    whitenas = pd.DataFrame(d)
    whitenas.to_csv('statescatter.csv')

