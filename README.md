# Capstone_Project1

## Bay Area Housing

This respository contains files and folders for my first capstone project of Springboard's Data Science Career Track course. It focuses on housing prices/cost of living in the Bay Area using Zillow data. See "Capstone_Project_1_Proposal.pdf" for more details.

The goal of this project is to develop a model that could reasonably predict the future housing market for the San Francisco Bay Area. This model could help visualize and influence how the market is impacted. If you'd like to find out more behind the motivation of this project, it could be found in the Capstone Project 1 Proposal document above.

This repository is laid out in the following structure:
- data
  - Zip_Zhvi_AllHomes.csv
    - Original time series data pulled from Zillow on average price per home by zip codes for the entire U.S.
  - Zip_MedianListingPricePerSqft_AllHomes.csv
    - Median price per square foot on all homes ordered by zip codes for the entire U.S.
  - by_zip.csv
    - Time series data on Bay Area housing prices grouped by zip codes
  - zip_id.csv
    - A csv file containing which cities belong to which Bay Area county
- Data_Wrangling
  - get_results.py
    - Code to extract Bay Area housing data from original Zillow data file 'Zip_Zhvi_AllHomes.csv'
- Data_Story
  - Data_Story.ipynb
    - This notebook utilizes the data to illustrate the motivation of this project. It compares the mean prices and price per square foot of Bay Area homes by zip codes to other zip codes around the U.S.
  - data_vis.py
    - Code that plots the time series data of counties around the Bay Area and U.s. Same code/graphs can be found in the notebook.
- EDA
  - Exploratory_Data_Analysis.ipynb
    - This notebook does some exploratory and statistical analysis on the data, mainly comparing Bay Area prices to the rest of the U.S.
  - EDA.py
    - This code is a direct iteration of the Jupyter notebook
  - Exploratory_Data_Analysis_Report.pdf
    - Report describing the analysis behind the Jupyter notebook and the findings.
- Machine_Learning
  - Capstone1_Machine_Learning.ipynb
    - Notebook building ARIMA models for each Bay Area county and the preprocessing necessary to build those models
  - Capstone1_Machine_Learning.py
    - Python file converted directly from the Jupyter notebook.
  - Machine Learning Report.pdf
    - Report summarizing what was done to each model. Details how each model performed and what could be done to improve upon them.
- Final
  - Capstone_Project1_Final_Report.pdf
    - This report summarizes what was done throughout the entire project step by step.
  - Capstone_Project1_Milestone_Report.pdf
    - In this report, every step taken up until the exploratory data analysis phase is laid out and describes how and why those steps were taken.
  - Capstone_Project1_Presentation.ipynb
    - Jupyter notebook to create a slideshow outlining the entire project
  - Capstone_Project1_Presentation.slides.html
    - Slideshow presentation summarizing the entire project
