### Data was obtained from via Zillow Research Data's Home Values
### (https://www.zillow.com/research/data/)
### This represents the median estimated home value across a given 
### region. I've pulled the data for all homes (SFR, condos/co-op) by
### zip code and saved it as 'Zip_Zhvi_AllHomes.csv'

# Import necessary libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime


# Read the downloaded .csv file into a Pandas DataFrame
df = pd.read_csv('Zip_Zhvi_AllHomes.csv', parse_dates=True)

# RegionName is used to list which zip code for each region
# Extract only the zip codes in 'CA' to narrow the list and reduce any
# redundancies when extracting the CountyName
ca_zip = df[df['State'] == 'CA']

### Create a list of Bay Area counties and use to extract them from CountyName
# Assign URL to a variable
url = 'https://mtc.ca.gov/about-mtc/what-mtc/nine-bay-area-counties'

# Request contents of URL into HTML format
r = requests.get(url)
text = r.text

# Apply BeautifulSoup
soup = BeautifulSoup(text, 'html.parser')

# After scrapping the page for the contents of the list of counties, I found
# they were under the body <div class="field field-name...."> (typed below)
county_list = soup.find(class_='field field-name-body field-type-text-with-' +
                        'summary field-label-hidden')

# Within the body the list is located with labels '<li>
county_items = county_list.find_all('li')

# Initialize a list to which extract the county names from HTML format and
# store them
counties = [county.contents[0] for county in county_items]
    

### Use the newly created list to extract those counties from df['CountyNames']
bayarea_zip = ca_zip[ca_zip['CountyName'].isin(counties)]


### Remove unnecessary columns like RegionID, SizeRank, State
del bayarea_zip['RegionID'], bayarea_zip['SizeRank'], bayarea_zip['State']

# Set the index to be the zip codes
bayarea_zip = bayarea_zip.set_index('RegionName')


# Create a new Data Frame that only contain columns City:CountyName and export
# to zip_id.csv to possible use for later analysis
zip_id = bayarea_zip.loc[:,'City':'CountyName']
zip_id.to_csv('zip_id.csv')


# Create a new Data Frame that only contain the price by month
zip_dates = bayarea_zip.loc[:,'1996-04':]

# Convert the column names in zip_dates to datetime objects
zip_dates.columns = pd.to_datetime(zip_dates.columns)

# Pivot by_zip to swap index and column names
by_zip = zip_dates.transpose()

# Export zip_by_date to a csv file
by_zip.to_csv('by_zip.csv')