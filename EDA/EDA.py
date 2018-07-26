import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from itertools import combinations



# import by_zip and transform it
df = pd.read_csv('C:\\Users\\Curtis\\Desktop\\Springboard\\Capstone_Project1\\data\\Zip_Zhvi_AllHomes.csv',
                   parse_dates = True)
df = df.fillna(0)
df = df.set_index('RegionName')
full_size = df.drop(['RegionID', 'Metro', 'City'], axis = 1)
full = df.drop(['RegionID', 'Metro', 'City', 'SizeRank'], axis = 1)



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
united_states = full[~((full['CountyName'].isin(bayarea_counties)) & 
                (full['State'] == 'CA'))]
us = united_states.loc[:,'2010-01':'2018-04']


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


# Calculate percentage of data has 'inf' change
n_ba_pc_inf = (ba_pc['change'] == float('inf')).sum()
n_us_pc_inf = (us_pc['change'] == float('inf')).sum()

print(float(n_ba_pc_inf)/len(ba_pc), float(n_us_pc_inf)/len(us_pc))


# Remove data with 'inf' change
ba_pc = ba_pc[ba_pc['change'] != float('inf')]
us_pc = us_pc[us_pc['change'] != float('inf')]


# Plot distribution of Bay Area percent change
_ = plt.hist(ba_pc['change'], bins = 10)
_ = plt.title('Distribution of Percent Change of Bay Area (2010-2018)')
_ = plt.xlabel('Percent Change (%)')
_ = plt.ylabel('Count')
_ = plt.xlim([-0.6, 2.5])


# Plot distribution of U.S. percent change
_ = plt.hist(us_pc['change'], bins = 20)
_ = plt.title('Distribution of Percent Change of U.S. (2010-2018)')
_ = plt.xlabel('Percent Change (%)')
_ = plt.ylabel('Count')
_ = plt.xlim([-0.6, 2.5])


# Calculate z statistic and p-value of percent change between Bay Area and U.S.
z_pc = (np.mean(ba_pc['change']) - np.mean(us_pc['change']))/np.sqrt((np.var(ba_pc['change'])/len(ba_pc['change']))+((np.var(us_pc['change'])/len(us_pc['change']))))
p_val_pc = stats.norm.sf(z_pc)


# Count how many zip codes are in each Bay Area county
bay_area.loc[:,['CountyName','2018-04']].groupby('CountyName').count()


# Plot distribution of median house prices by zip code by Bay Area county
plt.subplots(figsize = (16, 81))
for county in bayarea_counties:
    _ = plt.subplot(9, 1, bayarea_counties.index(county) + 1)
    _ = plt.hist(bay_area[bay_area['CountyName'] == county]['2018-04'])
    _ = plt.title('Distribution of Median Housing Price by Zip Code (' + county + ')')
    _ = plt.xlabel('Median Housing Price')
    _ = plt.ylabel('Count')
    _ = plt.xlim([300000,7000000])


# Initialize empty dictionary to eventually transform into a data frame
ba_county_ttest = {}


# Create for loop to calculate t-statistic and p-value between Bay Area counties
# and their median housing price by zip code
for comb1, comb2 in combinations(bayarea_counties, 2):
    ba_county_ttest[comb1, comb2] = stats.ttest_ind(bay_area[bay_area['CountyName'] == comb1]['2018-04'],
                                                   bay_area[bay_area['CountyName'] == comb2]['2018-04'])

# Transform data into a data frame and add if p-value is less than a 0.01 alpha
ba_ttest_df = pd.DataFrame(ba_county_ttest).transpose()
ba_ttest_df.columns = ['stat', 'p_value']
ba_ttest_df['p < alpha'] = ba_ttest_df['p_value'] < 0.01


# Create empty data frame
ba_mean_ci = {}

# Compute the confidence interval of the mean housing price for each Bay Area county
for county in bayarea_counties:
    mean, var, std = stats.bayes_mvs(bay_area[bay_area['CountyName'] == county]['2018-04'], alpha=0.99)
    ba_mean_ci[county] = int(mean[0]), int(mean[1][0]), int(mean[1][1])


ba_ci_df = pd.DataFrame(ba_mean_ci)
ba_ci_df = ba_ci_df.transpose()
ba_ci_df.columns = ['center', 'lower bound', 'upper bound']


# Categorize 'upper' and 'lower' tier counties
upper = ['Marin', 'San Francisco', 'San Mateo', 'Santa Clara']
lower = ['Alameda', 'Contra Costa', 'Napa', 'Sonoma']

# Slice bay_area data frame by 'upper' and 'lower'
ba_upper = bay_area[bay_area['CountyName'].isin(upper)]
ba_lower = bay_area[bay_area['CountyName'].isin(lower)]


# Perform t-test on 'upper' and 'lower' tier counties
t_ba_tiers, p_val_ba_tiers = stats.ttest_ind(ba_upper['2018-04'], ba_lower['2018-04'], equal_var = False)


# Create dataframe with Bay Area county names and percent change from 2010-01 to 2018-04
ba_pc_county = pd.concat([bay_area['CountyName'], ba_pc], axis = 1, join = 'inner')
ba_pc_county = ba_pc_county.loc[:, ['CountyName', 'change']]


# Calculate the mean percent change in each county
ba_pc_county.groupby('CountyName').mean()


# Plot the distribution of percent change per county by zip code
plt.subplots(figsize = (16, 81))
for county in bayarea_counties:
    _ = plt.subplot(9, 1, bayarea_counties.index(county) + 1)
    _ = plt.hist(ba_pc_county[ba_pc_county['CountyName'] == county]['change'])
    _ = plt.title(county)
    _ = plt.xlabel('Percent Change in Housing Price (%)')
    _ = plt.ylabel('Count')
    _ = plt.xlim([0.1, 2.2])


# Initialize an empty data frame
ba_pc_county_ttest = {}


# Create for loop to calculate t-statistic and p-value between Bay Area counties
# and their mean percent change by zip code
for comb1, comb2 in combinations(bayarea_counties, 2):
    ba_pc_county_ttest[comb1, comb2] = stats.ttest_ind(ba_pc_county[ba_pc_county['CountyName'] == comb1]['change'],
                                                   ba_pc_county[ba_pc_county['CountyName'] == comb2]['change'])

# Transform data into a data frame and add if p-value is less than a 0.01 alpha
ba_pc_ttest_df = pd.DataFrame(ba_pc_county_ttest).transpose()
ba_pc_ttest_df.columns = ['stat', 'p_value']
ba_pc_ttest_df['p < alpha'] = ba_pc_ttest_df['p_value'] < 0.01


