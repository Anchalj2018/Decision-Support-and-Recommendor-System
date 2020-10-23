import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
import json
import ast

spark = SparkSession.builder.appName('Business_analysis').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext
from  pyspark.sql.functions import regexp_replace,col,to_date



class BusinessAnalysis:
    
    def __init__(self, input):
         # to get all business into a dataframe
        self.df_business = spark.read.parquet(input).cache()
        self.df_business_attributes = spark.read.parquet(input + 'Attributes')
        self.df_categories = spark.read.parquet(input + 'Categories')

    # main function
    def dist_Cusines_province(self,output,prv):
    # read main input file        
        df_business=self.df_business
    
        df_cat=df_business.select(df_business["BusinessID"],df_business["State"],functions.explode(functions.split(df_business["Categories"],',')))
    
        df_cat.createOrReplaceTempView('categorytbl')
    
        df_cat_clean=spark.sql("select BusinessID,State,regexp_replace(col,' ','') as \
                    Categories from categorytbl where State='"+prv+"'")
    
        df_cat_clean.createOrReplaceTempView('cattbl')
    
        df_categories_province=spark.sql("select count(BusinessID) as BusinessCount,State,Categories \
                      from cattbl \
                      where Categories in ('Canadian(New)',\
				'Chinese',\
				'American(Traditional)',\
				'Italian',\
				'Mexican',\
				'MiddleEastern',\
				'Japanese',\
				'Indian',\
				'Mediterranean',\
				'AsianFusion',\
				'French',\
				'American(New)',\
				'Vietnamese',\
				'Korean',\
				'Taiwanese',\
				'Caribbean',\
				'Greek',\
				'Filipino',\
				'LatinAmerican',\
				'Lebanese',\
				'Portuguese',\
				'Pakistani',\
				'Hawaiian',\
				'Persian/Iranian',\
				'British',\
				'African',\
				'ModernEuropean',\
				'Afghan',\
				'Turkish',\
				'Irish',\
				'Arabian',\
				'Belgian',\
				'German',\
				'Spanish',\
				'Brazilian',\
				'Himalayan/Nepalese',\
				'International',\
				'Russian',\
				'Colombian',\
				'SriLankan',\
				'Ukrainian',\
				'Venezuelan',\
				'Argentine',\
				'Australian',\
				'Ethiopian',\
				'Hungarian',\
				'Malaysian',\
				'Singaporean',\
				'SouthAfrican',\
				'Bangladeshi',\
				'Egyptian',\
				'Mongolian') \
                        group by Categories,State order by BusinessCount desc limit 10")
    
        #Write Output to csv files
        df_categories_province.write.csv(output, header=True)
    
    def dist_buss_toronto_stars(self, output):
        df_business = self.df_business

        df_business.createOrReplaceTempView('business')
       
        df_business_clean=spark.sql("select *,regexp_replace(PostalCode,' ','') as ZipCode from business")

        df_toronto_data = df_business_clean.select(df_business_clean['BusinessID'], df_business_clean['Name'],\
                                            df_business_clean['ZipCode'],df_business_clean['Latitude'],\
                                           df_business_clean['Longitude'],functions.floor(df_business_clean['BusinessStars']).alias('Stars'))\
                          .where(df_business_clean['City']=='Toronto')

        
        # write data to output
        df_toronto_data.write.csv(output,header=True)

    def gen_business_categories(self, count, output):
        # group by categories 
        df_all_categories = self.df_categories
        
        df_categories_group = df_all_categories.groupBy(df_all_categories['Category'])
        
        # count of each group 
        df_categories_count = df_categories_group.agg(functions.count(df_all_categories['BusinessID']).alias('CategoryCount'))
        
        # sort groups
        df_categories_sorted = df_categories_count.orderBy(df_categories_count['CategoryCount'], ascending=False)
        
        # take top n rows
        df_top_categories = df_categories_sorted.limit(count)
        
        # write data to output
        df_top_categories.write.csv(output, header=True)

    def gen_cities_rest_counts(self, output):
        df_business = self.df_business

        # group by state and city
        df_group = df_business.groupBy(df_business['State'], df_business['City'])

        # get count of each group
        df_count = df_group.agg(functions.count(df_business['BusinessID']).alias('BusinessCount'))

        # sort data
        df_sorted = df_count.orderBy(df_count['BusinessCount'], ascending=False)

        # write to output
        df_sorted.write.csv(output, header=True)

    def gen_province_stars_avg(self, output):
        df_business = self.df_business

        # group by state
        df_group = df_business.groupBy(df_business['State'])
                
        # get average of stars in each state
        df_count = df_group.agg(functions.avg(df_business['BusinessStars']).alias('AverageStar'))

        # sort data
        df_sort = df_count.orderBy(df_count['State'])

        df_sort.write.csv(output, header=True)

    def gen_attr_main_distribution(self, attr_key):
        df_b_attr = self.df_business_attributes

        # filter only required attr_key
        df_filtered = df_b_attr.where(df_b_attr['attr_key'].contains(attr_key))

        # group by value of key
        df_group = df_filtered.groupBy(df_filtered['attr_val'].alias(attr_key))

        # get count of attributes 
        df_count = df_group.agg(functions.count(df_filtered['BusinessID']))

        # write data to output 
        df_count.write.csv('Business' + attr_key, header=True)

    def gen_attr_subset_distribution(self, attr_key):
        df_b_attr = self.df_business_attributes

        # filter only required attribute key for those who has True value
        df_filtered = df_b_attr.where((df_b_attr['attr_key'].contains(attr_key)) & (df_b_attr['attr_val'] == 'True'))

        # split column (Ambience_Casual) to two columns (Ambience, Casual)
        split_col = functions.split(df_filtered['attr_key'], '_')

        # include columns
        df_cols = df_filtered.withColumn('attr_name', split_col.getItem(0)) \
                             .withColumn('attr_value',  split_col.getItem(1))

        # select only required columns
        df_final = df_cols.select(df_cols['BusinessID'], df_cols['attr_name'], df_cols['attr_value'])

        # group data based on value
        df_group = df_final.groupBy(df_final['attr_value'].alias(attr_key))

        # get count of each group
        df_count = df_group.agg(functions.count(df_final['BusinessID']).alias('Count'))

        # write data to output
        df_count.write.csv('Business' + attr_key, header=True)


# main function
def main(buss_input,output_folder):
    buss_analysis = BusinessAnalysis(buss_input)
    buss_analysis.dist_buss_toronto_stars(output_folder + '/BusinessTorontoStars')
    buss_analysis.dist_Cusines_province(output_folder + '/BusinessCusinesProvince_AB','AB')
    buss_analysis.dist_Cusines_province(output_folder + '/BusinessCusinesProvince_ON','ON')
    buss_analysis.dist_Cusines_province(output_folder + '/BusinessCusinesProvince_QC','QC')
    buss_analysis.gen_business_categories(10, output_folder + 'BusinessTopTenCategories')
    buss_analysis.gen_cities_rest_counts(output_folder + 'BusinessCities')
    buss_analysis.gen_province_stars_avg(output_folder + 'BusinessProvinceAvgStars')
    
if __name__ == '__main__':
    buss_input=sys.argv[1]
    output_folder=sys.argv[2]
    main(buss_input,output_folder)

