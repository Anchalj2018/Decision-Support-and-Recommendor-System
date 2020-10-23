import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
import json
import ast

spark = SparkSession.builder.appName('User').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

# main function
def main(review_input,user_input,user_output):
    # read main input file
    df_review = spark.read.parquet(review_input)
    
    df_user=spark.read.json(user_input)
    
    df_review.createOrReplaceTempView('review')
    
    df_user.createOrReplaceTempView('user')
        
    df_user_filter=spark.sql("Select \
                                user_id as UserID,\
                                name as UserName, \
                                useful as Useful, \
                                funny as Funny, \
                                cool as Cool, \
                                review_count as ReviewCount, \
                                yelping_since as YelpingSince, \
                                average_stars as AverageStars \
                            from user \
                            where user_id in (select UserID from review)")

    #Write Output to parequet files
    df_user_filter.write.parquet(user_output)
    

if __name__ == '__main__':
    review_input=sys.argv[1]
    user_input=sys.argv[2]
    user_output =sys.argv[3]
    main(review_input,user_input,user_output)


