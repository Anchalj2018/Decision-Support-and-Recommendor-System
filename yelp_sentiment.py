import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+


from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('sentiment code').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml import Pipeline, PipelineModel
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer  
import string
import re
import string
import re

def strip_non_ascii(data_str):
      
    stripped = (c for c in data_str if 0 < ord(c) < 127)
    return ''.join(stripped)
strip_non_ascii_udf = udf(strip_non_ascii, StringType())




def remove_features(data_str):
    # compile regex
    url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
    punc_re = re.compile('[%s]' % re.escape(string.punctuation))
    num_re = re.compile('(\\d+)')

    alpha_num_re = re.compile("^[a-z0-9_.]+$")

    # remove hyperlinks
    data_str = url_re.sub(' ', data_str)
    
    # remove puncuation
    data_str = punc_re.sub(' ', data_str)
    # remove numeric 'words
    data_str = num_re.sub(' ', data_str)
    
    return(data_str)
remove_features_udf = udf(remove_features, StringType())


#join tokens
def join_text(x):
    joinedTokens_list = []
    x = " ".join(x)
    return x
join_udf = udf(lambda x: join_text(x), StringType())



def sentiment_analyzer_scores(sentence):
    analyzer = SentimentIntensityAnalyzer() 

    text = sentence[3]
    score = analyzer.polarity_scores(text)
    return(sentence[0], sentence[1],sentence[2],score['neg'],score['neu'],score['pos'],score['compound'])



def main(review_table,business_tabel,output_folder):
    
    #strip_non_ascii_udf = udf(strip_non_ascii, StringType())

    review_df = spark.read.parquet(review_table)
    review_text_lower = review_df.select(review_df['BusinessID'], functions.lower(review_df['Review']).alias("Review")) 
    review_text_lower.createOrReplaceTempView("review_tb")

    
    business_df = spark.read.parquet(business_tabel)
    business_toronto=business_df.filter(business_df.City=="Toronto")
    
    business_toronto.createOrReplaceTempView("business")
    df_business_clean=spark.sql("select *,regexp_replace(PostalCode,' ','') as ZipCode from business")


    

    #select business reviews for toronto
    df_business_clean.createOrReplaceTempView("business_table")
    df_review_filter=spark.sql("SELECT a.BusinessID,b.Name,a.Review FROM review_tb a,business_table b WHERE a.BusinessID=b.BusinessID")

   

    #text preprocessing
    df = df_review_filter.withColumn('text_non_asci',strip_non_ascii_udf(df_review_filter['Review']))
    rm_df = df.withColumn('clean_text',remove_features_udf(df['text_non_asci']))



    regexTokenizer = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'clean_text', outputCol = 'token')
    stopWordsRemover = StopWordsRemover(inputCol = 'token', outputCol = 'no_stopword')
    my_pipeline = Pipeline(stages=[regexTokenizer, stopWordsRemover])
    reg_model=my_pipeline.fit(rm_df)
    reg_df=reg_model.transform(rm_df)


    reg_joined=reg_df.withColumn("fine_text", join_udf( reg_df['no_stopword']))
    clean_df=reg_joined.select('BusinessID','Name','Review','fine_text')
        


    #calculate sentiment score
    review_rdd = rm_df.rdd
    score_rdd = review_rdd.map(sentiment_analyzer_scores)
    review_sentiment = spark.createDataFrame(score_rdd).cache()


    review_sentiment= review_sentiment.withColumnRenamed('_1', 'BusinessId')
    review_sentiment= review_sentiment.withColumnRenamed('_2', 'Business_Name')
    review_sentiment= review_sentiment.withColumnRenamed('_3', 'Review')
    review_sentiment= review_sentiment .withColumnRenamed('_4', 'Negative')
    review_sentiment = review_sentiment.withColumnRenamed('_5', 'Neutral')
    review_sentiment = review_sentiment.withColumnRenamed('_6', 'positive')
    review_sentiment = review_sentiment.withColumnRenamed('_7', 'Over_all')

    review_sentiment.coalesce(1).write.csv(output_folder + 'Sentiment_for_alltext',header=True)
    
    review_for_business = review_sentiment.groupBy('BusinessId')
    review_df = review_for_business.agg(round(functions.avg('Negative'),2).alias('avg_neg'),round(functions.avg('Neutral'),2).alias('avg_neu'),round(functions.avg('positive'),2).alias('avg_pos'),round(functions.avg('Over_all'),2).alias('avg_composite_score'))

    #get the business names
    review_df.createOrReplaceTempView('comp_table')
    new=spark.sql("select ct.*,bt.Name,bt.ZipCode,bt.Latitude,bt.Longitude from comp_table ct,business_table bt where bt.BusinessID=ct.BusinessID")
    new.show(2)

    new.write.csv(output_folder + '/BusinessSentiment', header=True)

if __name__ == '__main__':
    review_table=sys.argv[1]
    business_table=sys.argv[2]
    output_folder=sys.argv[3]
    main(review_table,business_table,output_folder)

