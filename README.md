# reddit-echo-chambers
Code repository presented in support of the PhD thesis 'Echoing Within and Between: Quantifying Echo Chamber Behaviours on Reddit'

## credentials
An empty folder which will need two credentials files for the code to run:
- *gcs_service_account.json*: with private account details for Google Cloud Service 
- *praw.json*: with private login details for the PRAW package

## resources
- *subreddit_labels_2019_01*: list of the 1000 most active subreddits by author count in January 2019 and their high-level topic labels
- *political_subreddit_descriptions*: descriptions of the political subreddits taken from their sidebars

## scripts
The python and R script files to run the data collection, analysis, and visualisation conducted in this research.

- *collection.py*: collects data from BigQuery and stores it on Google Cloud Storage
- *network.py*: network-level analysis presented in Chapter 5
- *plot_graphs.R*: an R script for plotting network graphs
- *plotting.py*: prepares the non-network visuals presented in the thesis
- *subreddits.py*: subreddit-level analysis presented in Chapter 4
- *tools.py*: general tool functions

## unfiled
- *newpyenv*: runs a shell script which creates the environment required to run the code in this repository.