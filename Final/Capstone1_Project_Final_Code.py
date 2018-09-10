import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# import by_zip and transform it
full = pd.read_csv('C:\\Users\\Curtis\\Desktop\\Springboard\\Capstone_Project1\\data\\Zip_Zhvi_AllHomes.csv',
                   parse_dates = True)


del full['RegionID'], full['SizeRank'], full['Metro'], full['City']

grouped = full.groupby(['State','CountyName']).mean()
grouped = grouped.iloc[:,1:]
grouped.columns = pd.to_datetime(grouped.columns)

## create a list of bay area counties
#bayarea_counties = ['Alameda', 'Contra Costa', 'Marin', 'Napa',
#                    'San Francisco', 'San Mateo', 'Santa Clara',
#                    'Solano', 'Sonoma']
##i = 1
##while i <= len(bayarea_counties):
##    plt.subplot(3, 3, i)
##    plt.plot(grouped[bayarea_counties[i-1]], label = bayarea_counties[i-1])
##    i += 1
##plt.axis([1996,2019,0,2000000])    
##plt.show()    
#        
#
#bayarea = grouped[bayarea_counties]
#bayarea['bay'] = bayarea.mean(axis=1)
#
#
##sns.stripplot(data = grouped.loc['2018-04',:].transpose())
##sns.stripplot(data = bayarea.loc['2018-04',:].transpose(), color = 'red')
#
## import price per square foot of all zips in u.s.
#ppsqft = pd.read_csv('Zip_MedianListingPricePerSqft_AllHomes.csv')
#
#del ppsqft['SizeRank'], ppsqft['Metro'], ppsqft['City']
#
#ppsqft_grouped = ppsqft.groupby('CountyName').mean().transpose()
#ppsqft_grouped = ppsqft_grouped.iloc[1:,:]
#ppsqft_grouped.index = pd.to_datetime(ppsqft_grouped.index)
#
#
#pct_change = pd.concat([grouped['2010-01'], grouped['2018-04']]).transpose()
#
#pct_change['change'] = (pct_change.iloc[:,1] - pct_change.iloc[:,0])/(pct_change.iloc[:,0])
#
#
#ppsqft_pct_change = pd.concat([ppsqft_grouped['2010-01'], ppsqft_grouped['2018-04']]).transpose()
#
#ppsqft_pct_change['change'] = (ppsqft_pct_change.iloc[:,1] - ppsqft_pct_change.iloc[:,0])/(ppsqft_pct_change.iloc[:,0])
#
#sns.swarmplot(pct_change['change'], orient = 'v')
#sns.swarmplot(pct_change['change'][bayarea_counties], orient = 'v', color = 'red')
#
#
#gt = grouped.transpose()
#ppsf_gt = ppsqft_grouped.transpose()
#
### add column to price_per_sqft indicating which zips are in the bay area
###price_per_sqft['in_bayarea'] = price_per_sqft['CountyName'].isin(bayarea_counties)
##
##
### pivot price_per_sqft to aggregate avg per county
##avg_price_sqft_county = price_per_sqft.pivot_table(index = 'CountyName',
##                                                   values = price_per_sqft.loc[:,'2010-01':].columns,
##                                                   aggfunc = 'mean')
##
##avg_price_sqft_county = avg_price_sqft_county.transpose()
##
##plt.plot(avg_price_sqft_county[bayarea_counties])
##
##
##ba_zip_avg = combined.pivot_table(index = combined.loc[:,'1996-04-01':'2018-04-01'].columns,
##                                  columns = 'CountyName',
##                                  aggfunc = 'mean')