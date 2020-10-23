import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types

import ast
import numpy as np
import csv
from operator import add
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import RegexTokenizer, CountVectorizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import IDF
from pyspark.sql.functions import udf
from pyspark.ml.clustering import LDA
from pyspark.ml import Pipeline, PipelineModel
from matplotlib import pyplot as plt
from wordcloud import WordCloud


spark = SparkSession.builder.appName('Topic modelling').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext



def main(review_table,business_table,output_folder):


    #Read reviews and business data
    review_df = spark.read.parquet(review_table)
    review_df.createOrReplaceTempView("reviews_table")

    business_df = spark.read.parquet(business_table)
    business_toronto=business_df.filter(business_df.City=="Toronto")
    business_toronto.createOrReplaceTempView("business_table")

    #collect reviews for each business
    business_review=spark.sql( """ SELECT BusinessID, collect_set(Review) AS total_review FROM reviews_table GROUP BY BusinessID """ )

    #convert reviews in string format
    merge_review = udf(lambda total_review: (" ").join(total_review))
    business_concat_review=business_review.withColumn("comb_review", merge_review(business_review['total_review'])).drop(business_review['total_review'])
    business_concat_review.createOrReplaceTempView("comb_review_table")

    #Keep reviews for business in toronto
    Reviews_for_business=spark.sql(""" SELECT c.BusinessID,b.Name AS BusinessName,b.BusinessStars,c.comb_review FROM comb_review_table AS c INNER JOIN business_table AS b ON c.BusinessID=b.BusinessID """)

    #pipleine to preprocess text data
    regexTokenizer = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'comb_review', outputCol = 'token')
    stopWordsRemover = StopWordsRemover(inputCol = 'token', outputCol = 'no_stopword')
    countVectorizer = CountVectorizer(inputCol="no_stopword", outputCol="rawcol")
    TDF = IDF(inputCol="rawcol", outputCol="idf_vec")
    text_pipeline = Pipeline(stages=[regexTokenizer, stopWordsRemover, countVectorizer, TDF])

    IDF_model = text_pipeline.fit(Reviews_for_business)
    #IDF_model.write().overwrite().save('IDF_model1')

    #collect the vacabulary from text  from count vectorizer model
    vocab=IDF_model.stages[2].vocabulary

    business_review_df=IDF_model.transform(Reviews_for_business)

    #two business categories base on low and high star rating
    reviews_low=business_review_df.where(business_review_df.BusinessStars<=3)
    reviews_high=business_review_df.where(business_review_df.BusinessStars>3)

    lda = LDA(k=6, seed=123, optimizer='online', featuresCol="idf_vec")
    vocab_word = udf(lambda termIndices: [vocab[idx] for idx in termIndices])

    #topic modelling on low rating business
    lowtopic_model = lda.fit(reviews_low)
    lowtopic_transform=lowtopic_model.transform(reviews_low)
    print("topic distribution for low rating business")
    lowtopic_transform.select('BusinessID','BusinessName','topicDistribution').show(4,False)
    #lowtopic_model.write().overwrite().save('lowtopic_model')
    
    #topic distribution
    low_dist=lowtopic_transform.withColumn('topic_distribution',lowtopic_transform['topicDistribution'].cast('string')).drop('topicDistribution')
    low_dist_df=low_dist.select('BusinessID','BusinessName','topic_distribution')    
    low_dist_df.write.csv(output_folder + '/Topic_low_business_topic_dist',header=True)
    
    #key topics
    lowreview_topics=lowtopic_model.describeTopics() 
    lowreview_topics_concat=lowreview_topics.withColumn("topic_word", vocab_word(lowreview_topics['termIndices']))
    
    
    low_df=lowreview_topics_concat.select('topic','topic_word')
    print("Topics for low rating business")
    low_df.show(6,False)
    low_df.coalesce(1).write.csv(output_folder + '/Topic_low_rating_topic',header=True)

    
    #topic modelling on high rating business
    high_topic_model = lda.fit(reviews_high)
    hightopic_transform=high_topic_model.transform(reviews_high)
    print("topic distribution for high rating business")
    hightopic_transform.select('BusinessID','BusinessName','topicDistribution').show(4,False)
    #high_topic_model.write().overwrite().save('high_topic_model')
    
    #topic distribution
    high_dist=hightopic_transform.withColumn('topic_distribution',hightopic_transform['topicDistribution'].cast('string')).drop('topicDistribution')
    high_dist_df=high_dist.select('BusinessID','BusinessName','topic_distribution')
    high_dist_df.write.csv(output_folder + '/Topic_high_business_topic_dist',header=True)

    #key topic 
    highreview_topics=high_topic_model.describeTopics()
    highreview_topics_concat=highreview_topics.withColumn("topic_word", vocab_word(highreview_topics['termIndices']))
    high_df=highreview_topics_concat.select('topic','topic_word')
    
    print("\nTopics for high rating business")
    high_df.show(6,False)
    high_df.coalesce(1).write.csv(output_folder + '/Topic_high_rating_topic',header=True)
    

if __name__ == '__main__':
    review_table=sys.argv[1]
    business_table=sys.argv[2]
    output_folder=sys.argv[3]
       
    main(review_table,business_table,output_folder)
