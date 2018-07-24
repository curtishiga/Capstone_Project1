import pandas as pd
import numpy as np
from datetime import datetime
import scipy.stats as stats



# import by_zip and transform it
df = pd.read_csv('Zip_Zhvi_AllHomes.csv',
                   parse_dates = True)
df = df.fillna(0)
full_size = df.drop(['RegionID', 'Metro', 'City'], axis = 1)
full = df.drop(['RegionID', 'Metro', 'City', 'SizeRank'], axis = 1)
full = full.set_index('RegionName')


# create a list of bay area counties
bayarea_counties = ['Alameda', 'Contra Costa', 'Marin', 'Napa',
                    'San Francisco', 'San Mateo', 'Santa Clara',
                    'Solano', 'Sonoma']


# Create Dataframe of Bay Area zip codes with county names
bay_area = full[(full['CountyName'].isin(bayarea_counties)) & 
                (full['State'] == 'CA')]


# Remove county names from bay_area
ba = bay_area.loc[:,'2010-01':'2018-04']


# Create Dataframe of non-Bay Area zip codes
us = full[~((full['CountyName'].isin(bayarea_counties)) & 
                (full['State'] == 'CA'))]
us = us.loc[:,'2010-01':'2018-04']


# Perform z-test
z = (np.mean(ba['2018-04']) - np.mean(us['2018-04']))/np.sqrt((np.var(ba['2018-04'])/len(ba['2018-04']))+((np.var(us['2018-04'])/len(us['2018-04']))))

p_val = stats.norm.sf(abs(z)) * 2

#
## 2010-2018 Percent Change
#

# Create new data frames with 2010 and 2018 data
ba_pc = pd.concat([ba['2010-01'], ba['2018-04']], axis = 1)
us_pc = pd.concat([us['2010-01'], us['2018-04']], axis = 1)


# Add percent change column to each new data frame
ba_pc['change'] = (ba['2018-04'] - ba['2010-01'])/ba['2010-01']
us_pc['change'] = (us['2018-04'] - us['2010-01'])/us['2010-01']


# Remove 'inf' change
ba_pc = ba_pc[ba_pc['change'] != float('inf')]
us_pc = us_pc[us_pc['change'] != float('inf')]


# Calculate the z score and p-value of the mean percent change
z_pc = (np.mean(ba_pc['change']) - np.mean(us_pc['change']))/np.sqrt((np.var(ba_pc['change'])/len(ba_pc['change']))+((np.var(us_pc['change'])/len(us_pc['change']))))
p_val_pc = stats.norm.sf(z_pc)

print(z_pc, p_val_pc)


#### Define a function to take bootstrap mean of a dataset
#def bs_rep_mean(data, size=1):
#    '''Takes a set of data and replicates the mean of resampled data'''
#    
#    reps = np.empty(size)
#    np.random.seed(20)
#    for i in range(size):
#        samples = np.random.choice(data, len(data))
#        reps[i] = np.mean(samples)
#    return reps
#
#
## Calculate the observed difference in means
#obs_diff = np.mean(ba['2018-04']) - np.mean(us['2018-04'])
#
#
## Take bootstrap replicate of mean on ba and us data
#ba_bs_mean = bs_rep_mean(ba['2018-04'], size = 10000)
#us_bs_mean = bs_rep_mean(us['2018-04'], size = 10000)
#
#
#stats.ttest_ind(ba_bs_mean, us_bs_mean, equal_var=False)