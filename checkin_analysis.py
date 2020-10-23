import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('Business_analysis').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext
from  pyspark.sql.functions import regexp_replace,col,to_date


class CheckinAnalysis:
    def __init__(self, input):
         # to get all business into a dataframe
        self.df_checkins = spark.read.parquet(input).cache()

    # generate heat map data for checkins
    def gen_checkin_heatmap(self, output):
        df_checkins = self.df_checkins

        # get day of week and hour
        df_day_hour = df_checkins.select(df_checkins['BusinessID'], \
                                    functions.date_format(df_checkins['CheckinDate'], 'HH').alias('CheckinHour'), \
                                    functions.date_format(df_checkins['CheckinDate'], 'u').alias('CheckinWeekDayNo'), \
                                    functions.date_format(df_checkins['CheckinDate'], 'E').alias('CheckinWeekDayName'))

        # group by day of week and hour
        df_groups = df_day_hour.groupBy(df_day_hour['CheckinWeekDayNo'], df_day_hour['CheckinWeekDayName'], df_day_hour['CheckinHour'])

        # get count of each group
        df_count = df_groups.agg(functions.count(df_day_hour['BusinessID']).alias('CheckinCount'))

        # order by week day
        df_sorted = df_count.orderBy(df_count['CheckinWeekDayNo'])

        # write to output
        df_sorted.write.csv(output, header=True)

    def gen_checkin_months(self, output):
        df_checkins = self.df_checkins

        # get months
        df_months = df_checkins.select(df_checkins['BusinessID'], \
                                        functions.date_format(df_checkins['CheckinDate'], 'MM').alias('MonthNo'), \
                                        functions.date_format(df_checkins['CheckinDate'], 'MMM').alias('MonthName'))

        # group by month no. and name
        df_group = df_months.groupBy(df_months['MonthNo'], df_months['MonthName'])

        # get count of each group
        df_count = df_group.agg(functions.count(df_months['BusinessID']).alias('CheckinCount'))

        # sort data by month no
        df_sorted = df_count.orderBy(df_count['MonthNo'])

        # write data to output
        df_sorted.write.csv(output, header=True)

    def gen_checkin_days(self, output):
        df_checkins = self.df_checkins

        # get checkin days
        df_days = df_checkins.select(df_checkins['BusinessID'], \
                                        functions.date_format(df_checkins['CheckinDate'], 'u').alias('DayNo'), \
                                        functions.date_format(df_checkins['CheckinDate'], 'E').alias('DayName'))

        # group by day no. and name
        df_group = df_days.groupBy(df_days['DayNo'], df_days['DayName'])

        # get count of each group
        df_count = df_group.agg(functions.count(df_days['BusinessID']).alias('CheckinCount'))

        # order by day no
        df_sorted = df_count.orderBy(df_count['DayNo'])

        # write output
        df_sorted.write.csv(output, header=True)

    def gen_checkin_hours(self, output):
        df_checkins = self.df_checkins

        # get checkin hours
        df_hours = df_checkins.select(df_checkins['BusinessID'], functions.date_format(df_checkins['CheckinDate'], 'HH').alias('Hour'))

        # group by hour
        df_group = df_hours.groupBy(df_hours['Hour'])

        # get count of each group
        df_count = df_group.agg(functions.count(df_hours['BusinessID']).alias('CheckinCount'))

        # order by hour
        df_sorted = df_count.orderBy(df_count['Hour'])

        # write data to output
        df_sorted.write.csv(output, header=True)


# main function
def main(checkin_input,output_folder):
    checkin_analysis = CheckinAnalysis(checkin_input)
    checkin_analysis.gen_checkin_months(output_folder + '/CheckinMonths')
    checkin_analysis.gen_checkin_days(output_folder + '/CheckinDays')
    checkin_analysis.gen_checkin_hours(output_folder + '/CheckinHours')
    checkin_analysis.gen_checkin_heatmap(output_folder + '/CheckinHeatMap')


if __name__ == '__main__':
    checkin_input=sys.argv[1]
    output_folder=sys.argv[2]
    main(checkin_input,output_folder)

