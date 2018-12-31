import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# import by_zip and transform it
full = pd.read_csv('../data/Zip_Zhvi_AllHomes.csv',
                   parse_dates = True)

del full['RegionID'], full['SizeRank'], full['Metro'], full['City']

grouped = full.groupby(['State', 'CountyName']).mean()
grouped = grouped.iloc[:,1:]
grouped.columns = pd.to_datetime(grouped.columns)

# create a list of bay area counties
bayarea_counties = ['Alameda', 'Contra Costa', 'Marin', 'Napa',
                    'San Francisco', 'San Mateo', 'Santa Clara',
                    'Solano', 'Sonoma']

for county in bayarea_counties:
    plt.plot(grouped.loc[('CA', county),:])
    plt.title(county)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()    
        

bayarea = grouped.loc['CA',:]
bayarea = bayarea.loc[bayarea.index.isin(bayarea_counties),:]


sns.stripplot(data = grouped.loc[:,'2018-04-01'])
sns.stripplot(data = bayarea.loc[:,'2018-04-01'], color = 'red')

# import price per square foot of all zips in u.s.
ppsqft = pd.read_csv('../data/Zip_MedianListingPricePerSqft_AllHomes.csv')

del ppsqft['SizeRank'], ppsqft['Metro'], ppsqft['City']

ppsqft_grouped = ppsqft.groupby('CountyName').mean()
ppsqft_grouped = ppsqft_grouped.iloc[:, 1:]
ppsqft_grouped.columns = pd.to_datetime(ppsqft_grouped.columns)


pct_change = pd.concat([grouped['2010-01-01'], grouped['2018-04-01']], axis = 1)

pct_change['change'] = (pct_change.iloc[:,1] - pct_change.iloc[:,0])/(pct_change.iloc[:,0])


ppsqft_pct_change = pd.concat([ppsqft_grouped['2010-01-01'], ppsqft_grouped['2018-04-01']], axis = 1)

ppsqft_pct_change['change'] = (ppsqft_pct_change.iloc[:,1] - ppsqft_pct_change.iloc[:,0])/(ppsqft_pct_change.iloc[:,0])

sns.swarmplot(pct_change['change'], orient = 'v')
sns.swarmplot(pct_change['change'][bayarea_counties], orient = 'v', color = 'red')