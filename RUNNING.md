# A Deeper Analysis of Restaurants using Yelp Dataset

# Requirements to run python files:

## Install required libraries
pip install matplotlib --user <br/>
pip install wordcloud --user <br/>
pip install nltk --user <br/>
pip install wikipedia --user <br/>

## Instructions to run

### Convert yelp json files to parquet
spark-submit business.py ../dataset/json/business.json ../dataset/json/province.json ../dataset/parquet/Business <br/>
spark-submit review.py ../dataset/parquet/Business ../dataset/json/review.json ../dataset/parquet/Review <br/>
spark-submit checkin.py ../dataset/parquet/Business ../dataset/json/checkin.json ../dataset/parquet/Checkin <br/>
spark-submit user.py ../dataset/parquet/Review ../dataset/json/user.json ../dataset/parquet/User <br/>

### Codes for data analysis
spark-submit business_analysis.py ../dataset/parquet/Business ../dataset/output/ <br/>
spark-submit checkin_analysis.py ../dataset/parquet/Checkin ../dataset/output/ <br/>
spark-submit user_analysis.py ../dataset/parquet/User ../dataset/output/ <br/>
spark-submit review_analysis.py ../dataset/parquet/Review ../dataset/output/ <br/>
spark-submit ethnicitydistribution_analysis.py ../dataset/parquet/Business ../dataset/postcode_toronto.csv ../dataset/wellbeing_ethnicity.csv/ <br/>
spark-submit labour_v_review_analysis.py ../dataset/parquet/Business ../dataset/wellbeing_labour.csv ../dataset/postcode_toronto.csv/ <br/>

### Codes for topic modelling
spark-submit topic_modelling.py ../dataset/parquet/Review ../dataset/parquet/Business ../dataset/output <br/>
spark-submit topicDistribution_explode.py ../dataset/output/Topic_low_business_topic_dist ../dataset/output/Topic_high_business_topic_dist ../dataset/output/ <br/>

### Codes for recommender part
spark-submit recommender_train.py ../dataset/output/BusinessSentiment ../dataset/parquet/User ../dataset/parquet/Review ../dataset/output/ <br/>
spark-submit recommender_test.py ../dataset/output/BusinessSentiment ../dataset/parquet/User ../dataset/parquet/Review ../dataset/output/TrainedModel ../dataset/output/ <br/>
