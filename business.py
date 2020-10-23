import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
import json
import ast

spark = SparkSession.builder.appName('Business').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

# define schema for business
SCHEMA = types.StructType([
    types.StructField("address", types.StringType()),
    types.StructField("attributes", types.StringType()),
    types.StructField("business_id", types.StringType()),
    types.StructField("categories", types.StringType()),
    types.StructField("city", types.StringType()),
    types.StructField("hours", types.StringType()),
    types.StructField("is_open", types.LongType()),
    types.StructField("latitude", types.DoubleType()),
    types.StructField("longitude", types.DoubleType()),
    types.StructField("name", types.StringType()),
    types.StructField("postal_code", types.StringType()),
    types.StructField("review_count", types.LongType()),
    types.StructField("stars", types.DoubleType()),
    types.StructField("state", types.StringType()),
])

# define a function to read nested json objects for attributes column

@functions.udf(returnType=types.StringType())
def flatten_attributes(input):
    output = ""

    if (input != None):
        attributes = json.loads(input)

        for k1, v1 in attributes.items():

            if (v1.startswith("{")):
                sub_values = ast.literal_eval(v1)
                for k2, v2 in sub_values.items():
                    output = output + k1 + "_" + k2 + ":" + str(v2) + "@"
            else:              
                output = output + k1 + ":" + str(v1) + "@"

    output=output.replace("u'", "")
    output=output.replace("\'","")
    return output


# main function
def main(buss_input,province_input,buss_output):
    # read main input file
    df_business = spark.read.json(buss_input, schema=SCHEMA)
    
    df_province=spark.read.json(province_input)
    
    df_business.createOrReplaceTempView('business')
    
    df_province.createOrReplaceTempView('province')
    
    #Business Table for Resturants which are open
    df_buss_filter=spark.sql("Select \
                                business_id as BusinessID, \
                                attributes as Attributes, \
                                name as Name, \
                                city as City, \
                                state as State, \
                                postal_code as PostalCode, \
                                latitude as Latitude, \
                                longitude as Longitude, \
                                stars as BusinessStars,\
                                categories as Categories,\
                                review_count as ReviewCount \
                            from business \
                            where is_open=1 \
                                and state in (select code from province) \
                                and postal_code is not null \
                                and stars is not null \
                                and latitude is not null \
                                and longitude is not null \
                                and review_count is not null \
                                and categories is not null ").cache()
    

    # Business Categories 

    # explode categories using ","
    df_categories = df_buss_filter.select(df_buss_filter['BusinessID'], functions.explode(functions.split(df_buss_filter['Categories'], ',')).alias('Category'))

    # trim empty spaces of category name
    df_trim_categories = df_categories.select(df_categories['BusinessID'], functions.regexp_replace(df_categories['Category'], ' ', '').alias('Category'))


    # get restaurants
    df_bus_restaurants = df_buss_filter.where(df_buss_filter['Categories'].contains('Restaurant') & df_buss_filter['Categories'].contains('Food'))

    # fetch required columns for Business Table
    df_bussiness_fil = df_bus_restaurants.select(df_bus_restaurants['BusinessID'],df_bus_restaurants['Name'],df_bus_restaurants['City'],\
                                               df_bus_restaurants['State'],df_bus_restaurants['PostalCode'],df_bus_restaurants['Latitude'],\
                                               df_bus_restaurants['Longitude'],df_bus_restaurants['BusinessStars'],df_bus_restaurants['Categories'],\
                                               df_bus_restaurants['ReviewCount'])

    # Business Attributes

    # flatten attributes column [because of nested objects] to a string with @ delimiter
    df_flat_attrs = df_bus_restaurants.select(df_bus_restaurants['BusinessID'], flatten_attributes(df_bus_restaurants['Attributes']).alias('Attributes'))

    # explode  attributes to rows grouped by business id
    df_exploded_attrs = df_flat_attrs.select(df_flat_attrs['BusinessID'], functions.explode(functions.split(df_flat_attrs['Attributes'], '@')).alias('Attributes'))

    # filter valid attributes that has value
    df_valid_attrs = df_exploded_attrs.where(df_exploded_attrs['Attributes'] != '')

    # split attribute using : {key:value} to two columns
    df_seperated_attrs = df_valid_attrs.withColumn('attr_key', functions.split(df_valid_attrs['Attributes'], ':')[0]) \
                                       .withColumn('attr_val', functions.split(df_valid_attrs['Attributes'], ':')[1])

    # select business id, attribute key and value
    df_fil_attrs = df_seperated_attrs.select(df_seperated_attrs['BusinessID'], df_seperated_attrs['attr_key'], df_seperated_attrs['attr_val']) \
                                        .filter((df_seperated_attrs['attr_val'] != 'None') & (df_seperated_attrs['attr_val'] != 'none'))
    
    #creating parquet files
    df_bussiness_fil.write.parquet(buss_output)
    df_fil_attrs.write.parquet(buss_output + 'Attributes')
    df_trim_categories.write.parquet(buss_output + 'Categories')


if __name__ == '__main__':
    buss_input=sys.argv[1]
    province_input=sys.argv[2]
    buss_output = sys.argv[3]
    main(buss_input,province_input,buss_output)
