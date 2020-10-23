import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('Review_analysis').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext
from  pyspark.sql.functions import regexp_replace,col,to_date


class ReviewAnalysis:
    def __init__(self, input):
         # to get all business into a dataframe
        self.df_reviews = spark.read.parquet(input).cache()

    # generate heat map data for checkins
    def gen_review_years(self, output):
        df_reviews = self.df_reviews
        
        # get year column from date
        df_columns = df_reviews.select(df_reviews['BusinessID'], functions.date_format(df_reviews['Date'], 'YYYY').alias('Year'))

        # group by year
        df_groups = df_columns.groupBy(df_columns['Year'])

        # get count of each year
        df_counts = df_groups.agg(functions.count(df_columns['BusinessID']).alias('ReviewCount'))

        # sort data by year
        df_sorted = df_counts.orderBy(df_counts['Year'])

        # write data to output
        df_sorted.write.csv(output, header=True)
        
    def gen_temporal_review_rating(self, output):
        df_reviews = self.df_reviews
        
        #group data to get count for reviews over date
        review_star = df_reviews.groupby('ReviewStars', 'Date').agg(functions.count('ReviewStars').alias("Count"))
        
        # write data to output
        df_sorted.write.csv(output, header=True)
        
        


# main function
def main(review_input,output_folder):
    review_analysis = ReviewAnalysis(review_input)
    review_analysis.gen_review_years(output_folder + '/ReviewYears')
    review_analysis.gen_temporal_review_rating(output_folder + '/ReviewOverYears')


if __name__ == '__main__':
    review_input=sys.argv[1]
    output_folder=sys.argv[2]
    main(review_input,output_folder)

