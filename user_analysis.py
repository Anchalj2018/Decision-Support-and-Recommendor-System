import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('UserAnalysis').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

# this is the user schema
USERS_SCHEMA = types.StructType([
    types.StructField("UserID", types.StringType()),
    types.StructField("UserName", types.StringType()),
    types.StructField("Useful", types.IntegerType()),
    types.StructField("Funny", types.IntegerType()),
    types.StructField("Cool", types.IntegerType()),
    types.StructField("ReviewCount", types.IntegerType()),
    types.StructField("YelpingSince", types.DateType()),
    types.StructField("AverageStars", types.FloatType()),
])


class UserAnalysis:
    RANGE_PERIOD = 10
    
    def __init__(self, input):
         # to get all users into a dataframe
        self.df_users = spark.read.parquet(input).cache()
        
    def gen_yelping_years(self, output):
        df_users = self.df_users

        # to group by users based on yelping start date
        df_group = df_users.groupBy(functions.year(df_users['YelpingSince']).alias('Year'))

        # aggregate count of usersbased on yelping start date
        df_group_count = df_group.agg(functions.count(df_users['UserId']).alias('UsersCount'))

        # sort data descending by year
        df_year_sorted = df_group_count.orderBy(df_group_count['Year'])
        
        # write data to output
        df_year_sorted.write.csv(output, header=True)
        
    def gen_review_counts(self, output):
        df_users = self.df_users

        # get division by user review count by 10
        df_range_div = df_users.select(df_users['UserID'], (df_users['ReviewCount'] / UserAnalysis.RANGE_PERIOD).cast(types.IntegerType()).alias('BaseRange'))

        # find start-end of range
        df_ranges = df_range_div.select( \
                                  (df_range_div['UserID']), \
                                  (df_range_div['BaseRange'] * UserAnalysis.RANGE_PERIOD).alias('StartRange'),  \
                                  (df_range_div['BaseRange'] * UserAnalysis.RANGE_PERIOD + (UserAnalysis.RANGE_PERIOD - 1)).alias('EndRange'))

        # group data by start , end range
        df_ranges_group = df_ranges.groupBy(df_ranges['StartRange'], df_ranges['EndRange'])

        # get count of users in each range
        df_ranges_count = df_ranges_group.agg(functions.count(df_ranges['UserID']).alias('UserCount'))

        # organize data -> columns, sort
        df_ranges_final = df_ranges_count.select( \
                        functions.concat(df_ranges_count['StartRange'], functions.lit('-'), df_ranges_count['EndRange']).alias('ReviewNo'), \
                        df_ranges_count['UserCount']) \
                        .orderBy(df_ranges_count['StartRange'])

        # write data to output
        df_ranges_final.write.csv(output, header=True)

    def gen_star_counts(self, output):
        df_users = self.df_users

        # get users stars base number
        df_stars_range = df_users.select(df_users['UserID'], functions.floor(df_users['AverageStars']).alias('Stars'))

        # group by user stars
        df_stars_groups = df_stars_range.groupBy(df_stars_range['Stars'])

        # get count of each group
        df_stars_count = df_stars_groups.agg(functions.count(df_stars_range['UserID']).alias('UsersCount'))

        # sort data
        df_sorted = df_stars_count.orderBy(df_stars_count['Stars'])
        
        # write data to output
        df_sorted.write.csv(output, header=True)

# main function
def main(user_input,output_folder):
    user_analysis = UserAnalysis(user_input)
    user_analysis.gen_yelping_years(output_folder + '/UserYelpingYears')
    user_analysis.gen_review_counts(output_folder + '/UserReviewCounts')
    user_analysis.gen_star_counts(output_folder + '/UserStarCounts')


if __name__ == '__main__':
    user_input=sys.argv[1]
    output_folder=sys.argv[2]
    main(user_input,output_folder)
