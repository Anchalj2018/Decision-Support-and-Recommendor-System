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
def main(buss_input,checkin_input,checkin_output):
    # read main input file
    df_business = spark.read.parquet(buss_input)
    
    # read checkins
    df_checkin=spark.read.json(checkin_input)
    
    # join and get checkins for businesses we want
    df_rest_checkins = df_checkin.join(df_business, df_business['BusinessID'] == df_checkin['business_id'], 'left_semi')

    # explode checkin dates to rows
    df_explode = df_rest_checkins.select(df_rest_checkins['business_id'],  functions.explode(functions.split(df_rest_checkins['date'], ',')).alias('date'))

    # rename columns 
    df_rename_columns = df_explode.select(df_explode['business_id'].alias('BusinessID'), df_explode['date'].alias('CheckinDate'))

    # cast string dates to DateTime
    df_cast_dates = df_rename_columns.select(df_rename_columns['BusinessID'], functions.from_unixtime(functions.unix_timestamp('CheckinDate', 'yyyy-MM-dd HH:mm:ss')).alias('CheckinDate'))

    # remove null dates
    df_valid_dates = df_cast_dates.where(df_cast_dates['CheckinDate'].isNotNull())

    # write Output to Parquet files
    df_valid_dates.write.parquet(checkin_output)
    

if __name__ == '__main__':
    bussiness_input=sys.argv[1]
    checkin_input=sys.argv[2]
    checkin_output = sys.argv[3]
    main(bussiness_input,checkin_input,checkin_output)
