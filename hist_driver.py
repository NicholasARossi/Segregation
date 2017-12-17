import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import entropy_estimators as EE

mpl.rcParams['pdf.fonttype']=42
sns.set_style('white')




if __name__== "__main__" :
    #import school data
    ### depending on how you do this you might get a memory error, using in a jupyter notebook works
    data = pd.read_csv("schools.csv",error_bad_lines=False, index_col=False, encoding="ISO-8859-1")

    #parse specific states
    MN_data = data.where(data['LEA_STATE'] == 'MN')
    MN_data = MN_data.dropna()
    GA_data = data.where(data['LEA_STATE'] == 'GA')
    GA_data = GA_data.dropna()
    MA_data = data.where(data['LEA_STATE'] == 'MA')
    MA_data = MA_data.dropna()

    MN_values = []
    temp_dat = MN_data

    for r in range(len(temp_dat)):
        white = temp_dat['SCH_ENR_WH_M'].iloc[r] + temp_dat['SCH_ENR_WH_F'].iloc[r]
        total = temp_dat['TOT_ENR_M'].iloc[r] + temp_dat['TOT_ENR_F'].iloc[r]
        MN_values.append(white / total)

    GA_values = []
    temp_dat = GA_data
    for r in range(len(temp_dat)):
        white = temp_dat['SCH_ENR_WH_M'].iloc[r] + temp_dat['SCH_ENR_WH_F'].iloc[r]
        total = temp_dat['TOT_ENR_M'].iloc[r] + temp_dat['TOT_ENR_F'].iloc[r]
        GA_values.append(white / total)

    plt.close('all')
    import entropy_estimators as EE
    sns.set_style('white')
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    bins=np.arange(-.1,1.1,.05)


    density, bins = np.histogram(MN_values, bins=bins)

    ax[0].plot(bins[1:]-.025, density/np.sum(density),  ls='steps',color='darkturquoise',linewidth=5,alpha=0.75)


    s = np.random.normal(.82, np.sqrt(.82*(1-.82)/100), 10000)

    density, bins = np.histogram(s, bins=bins)

    ax[0].plot(bins[1:]-.025, density/np.sum(density),  ls='steps',color='darkgrey',linewidth=5,alpha=0.75)


    density, bins = np.histogram(GA_values, bins=bins)

    ax[1].plot(bins[1:]-.025, density/np.sum(density),  ls='steps',color='magenta',linewidth=5,alpha=0.75)


    s = np.random.normal(.55, np.sqrt(.55*(1-.55)/100), 10000)

    density, bins = np.histogram(s, bins=bins)

    ax[1].plot(bins[1:]-.025, density/np.sum(density),  ls='steps',color='darkgrey',linewidth=5,alpha=0.75)

    ax[1].set_ylim([0,.5])
    ax[0].set_ylim([0,.5])
    ax[1].set_ylabel('Probability')
    ax[0].set_ylabel('Probability')
    ax[1].set_xlabel('Percent White')
    ax[0].set_xlabel('Percent White')


    plt.tight_layout()

    fig.savefig('figures/MN_GA_density.pdf')

    plt.close('all')

    sns.set_style('white')
    fig, ax = plt.subplots(1, 2, figsize=(6, 2))
    # bins=np.arange(-.1,1.1,.05)



    mn = EE.kldiv(EE.vectorize(np.random.normal(.82, np.sqrt(.82 * (1 - .82) / 100), 1000)), EE.vectorize(MN_values),
                  k=20)
    ga = EE.kldiv(EE.vectorize(np.random.normal(.55, np.sqrt(.55 * (1 - .55) / 100), 1000)), EE.vectorize(GA_values),
                  k=20)
    objects = [mn, ga]
    y_pos = np.arange(len(objects))

    ax[0].bar(y_pos * 2 + 1, objects, align='center', color=['darkturquoise', 'magenta'])
    # plt.xticks(y_pos, objects)
    # plt.ylabel('Usage')
    # plt.title('Programming language usage')
    ax[0].set_xticks(y_pos * 2 + 1)
    ax[0].set_xticklabels(['Minnesota', 'Georgia'])
    ax[0].set_xlim([0, 4])

    ax[1].bar(y_pos * 2 + 1, [9.2, 13.3], align='center', color=['darkturquoise', 'magenta'])
    # plt.xticks(y_pos, objects)
    # plt.ylabel('Usage')
    # plt.title('Programming language usage')
    ax[1].set_xticks(y_pos * 2 + 1)
    ax[1].set_xticklabels(['Minnesota', 'Georgia'])
    ax[1].set_xlim([0, 4])
    ax[1].set_ylabel('Average Z Score')
    ax[0].set_ylabel('Average Z Score')
    bbox_inches = 'tight'
    fig.savefig('figures/MN_GA_quant.pdf', bbox_inches='tight')


    #Massachucets schools
    MA_values = []
    temp_dat = MA_data

    for r in range(len(temp_dat)):
        white = temp_dat['SCH_ENR_WH_M'].iloc[r] + temp_dat['SCH_ENR_WH_F'].iloc[r]
        total = temp_dat['TOT_ENR_M'].iloc[r] + temp_dat['TOT_ENR_F'].iloc[r]
        final = white / total
        if final == 0:
            print(temp_dat['SCH_NAME'].iloc[r])
            print(temp_dat['TOT_ENR_M'].iloc[r])
        MA_values.append(final)

    plt.close('all')
    sns.set_style('white')
    fig, ax = plt.subplots()
    bins = np.arange(-.1, 1.1, .05)

    density, bins = np.histogram(MA_values, bins=bins)

    ax.plot(bins[1:] - .025, density / np.sum(density), ls='steps', color='darkturquoise', linewidth=5, alpha=0.75)

    s = np.random.normal(.75, np.sqrt(.75 * (1 - .75) / 100), 10000)

    density, bins = np.histogram(s, bins=bins)

    ax.plot(bins[1:] - .025, density / np.sum(density), ls='steps', color='darkgrey', linewidth=5, alpha=0.75)

    ax.set_ylim([0, .5])
    ax.set_ylabel('Probability')
    ax.set_xlabel('Percent White')


    plt.tight_layout()

    fig.savefig('figures/MA_results.pdf')

