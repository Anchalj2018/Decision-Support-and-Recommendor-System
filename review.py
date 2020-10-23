import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
import json
import ast

spark = SparkSession.builder.appName('Review_JsonToParquet').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext
from  pyspark.sql.functions import regexp_replace,col,to_date

# main function
def main(buss_input,review_input,review_output):
    # read main input file
    df_business = spark.read.parquet(buss_input)
    
    df_review=spark.read.json(review_input)
    
    df_business.createOrReplaceTempView('business')
    
    df_review.createOrReplaceTempView('review')
    #Remove /n from reviews text
    df_review_mod=df_review.withColumn("text", regexp_replace(df_review["text"], '\n',''))
    #removing null text rows
    df_review_mod_fil=df_review_mod.filter((df_review_mod['text'] != ' '))
        
    df_review_filter=spark.sql("SELECT \
                                    review.review_id AS ReviewID, \
                                    review.user_id AS UserID, \
                                    review.business_id AS BusinessID,  \
                                    review.date AS ReviewDate, \
                                    review.stars AS ReviewStars, \
                                    review.text AS Review, \
                                    review.useful \
                                FROM review  \
                                WHERE  review.business_id in (SELECT BusinessID FROM business)")
    
    df_review_fil=df_review_filter.withColumn("Date",to_date(col('ReviewDate')).alias('Date').cast("date")).drop(df_review_filter["ReviewDate"])
   
    #Write Output to Parquet files
    df_review_fil.write.parquet(review_output)
    

if __name__ == '__main__':
    buss_input=sys.argv[1]
    review_input=sys.argv[2]
    review_output = sys.argv[3]
    main(buss_input,review_input,review_output)
