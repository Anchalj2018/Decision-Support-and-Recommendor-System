import sys


from functools import reduce
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.types import StructField,IntegerType, StructType,StringType
spark = SparkSession.builder.appName('Ethnicity Analysis').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

def main(business_file, postcode_file, ethnicity_file):
    # load files
    df_business = spark.read.parquet(business_file)
    df_postcode = spark.read.csv(postcode_file, header=True)
    df_ethnicity = spark.read.csv(ethnicity_file, header=True)

    # filter for businesses in toronto
    df_toronto = df_business.where("City like '%Toronto%'")

    # Combine external wellbeing  datasets to yelp datasets and pre-process
    df_join = df_wellbeing.join(df_postcode, on=['Neighbourhood'], how='left')
    df_join = df_join.drop('Combined Indicators', 'Borough')
    new_cols_ethn = [c.strip(' ') for c in df_ethnicity.columns]
    old_cols_ethn = df_ethnicity.schema.names
    df_ethnicity = reduce(
        lambda df_ethnicity, idx: df_ethnicity.withColumnRenamed(old_cols_ethn[idx], new_cols_ethn[idx]),
        range(len(old_cols_ethn)), df_ethnicity)

    df_indian = df_toronto.where("Categories like '%Indian%'")
    df_ethnicity = df_ethnicity.join(df_postcode, on=['Neighbourhood'], how='left')
    df_ethnicity_small = df_ethnicity.select('Neighbourhood', 'Total Population','South Asian', 'Postcode')
    df_ethn_norm = df_ethnicity_small

    cols = ['Chinese', 'South Asian', 'Black', 'Filipino', 'Latin American', 'Southeast Asian', 'Arab',
            'West Asian', 'Korean', 'Japanese', 'Not a Visible Minority']

    for field in df_ethnicity_small.columns:
        if field in cols:
            df_ethn_norm = df_ethn_norm.withColumn(field, col(field) / col("Total Population"))

    df_indian_ethn = df_indian.withColumn("PostCode", functions.substring_index(col("PostalCode"), " ", 1)).join(
        df_ethn_norm, on='PostCode', how='left')

    df_ind_eth_sort = df_indian_ethn.orderBy('BusinessStars', ascending=False).select('BusinessID', 'Name', 'Latitude',
                                                                                      'Longitude', \
                                                                                      'BusinessStars', 'Neighbourhood',
                                                                                      'South Asian')

    df_ind_eth_sort.coalesce(1).write.csv('df_ind_eth_sort.csv')

if __name__ == '__main__':
    business_file = sys.argv[1]
    postcode_file = sys.argv[2]
    ethnicity_file = sys.argv[3]
    main(business_file, postcode_file, ethnicity_file)
