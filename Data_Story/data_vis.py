import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt



# import by_zip and transform it
full = pd.read_csv('Zip_Zhvi_AllHomes.csv',
                   parse_dates = True)


del full['RegionID'], full['SizeRank'], full['Metro'], full['City']

# full = full.set_index(['State', 'CountyName', 'RegionName'])

#full.index.name = 'Date'
#
#full = full.reset_index()
#
grouped = full.groupby('CountyName').mean().transpose()
grouped = grouped.iloc[1:,:]
grouped.index = pd.to_datetime(grouped.index)

# create a list of bay area counties
bayarea_counties = ['Alameda', 'Contra Costa', 'Marin', 'Napa',
                    'San Francisco', 'San Mateo', 'Santa Clara',
                    'Solano', 'Sonoma']
i = 1
while i <= len(bayarea_counties):
    plt.subplot(3, 3, i)
    plt.plot(grouped[bayarea_counties[i-1]], label = bayarea_counties[i-1])
    i += 1
plt.axis([1996,2019,0,2000000])    
plt.show()    
        

bayarea = grouped[bayarea_counties]
bayarea['bay'] = bayarea.mean(axis=1)


## import full zillow research data
#full_zillow = pd.read_csv('Zip_Zhvi_AllHomes.csv',
#                          index_col = 'RegionName')
#
#
## import price per square foot of all zips in u.s.
#price_per_sqft = pd.read_csv('Zip_MedianListingPricePerSqft_AllHomes.csv',
#                             index_col = 'RegionName')
#
#
## add column to price_per_sqft indicating which zips are in the bay area
##price_per_sqft['in_bayarea'] = price_per_sqft['CountyName'].isin(bayarea_counties)
#
#
## pivot price_per_sqft to aggregate avg per county
#avg_price_sqft_county = price_per_sqft.pivot_table(index = 'CountyName',
#                                                   values = price_per_sqft.loc[:,'2010-01':].columns,
#                                                   aggfunc = 'mean')
#
#avg_price_sqft_county = avg_price_sqft_county.transpose()
#
#plt.plot(avg_price_sqft_county[bayarea_counties])
#
#
#ba_zip_avg = combined.pivot_table(index = combined.loc[:,'1996-04-01':'2018-04-01'].columns,
#                                  columns = 'CountyName',
#                                  aggfunc = 'mean')