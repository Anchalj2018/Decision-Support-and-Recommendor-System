import sys
import pandas as pd

from functools import reduce
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.types import StructField,IntegerType, StructType,StringType
spark = SparkSession.builder.appName('Labour Analysis').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext


labour_schema = types.StructType([
    types.StructField('Neighbourhood', StringType()),
    types.StructField('Neighbourhood Id', StringType()),
    types.StructField('Combined Indicators', StringType()),
    types.StructField('Total Population', IntegerType()),
    types.StructField('Labour Force Category', IntegerType()),
    types.StructField('In Labour Force', IntegerType()),
    types.StructField('Unemployed', IntegerType()),
    types.StructField('Not in Labour Force', IntegerType()),
])

def main(business_data, labour_data, postcode_data):

    df_business = spark.read.parquet(business_data)
    df_labour = spark.read.csv(labour_data, header=True)
    df_postcode = spark.read.csv(postcode_file, header=True)

    df_toronto = df_business.where("City like '%Toronto%'")

    # Strip spaces from columns
    new_cols_lb = [c.strip(' ') for c in df_labour.columns]
    old_cols_lb = df_labour.schema.names
    df_labour = reduce(lambda df_labour, idx: df_labour.withColumnRenamed(old_cols_lb[idx], new_cols_lb[idx]),
                       range(len(old_cols_lb)), df_labour)

    df_labour = df_labour.join(df_postcode, on=['Neighbourhood'], how='left')
    df_labour = df_labour.drop('CombinedIndicators', 'Borough', 'TotalPopulation')
    df_lb_norm = df_labour

    df_lb_norm = df_lb_norm.withColumn("LabourForceCategory", df_lb_norm["LabourForceCategory"].cast(IntegerType()))
    df_lb_norm = df_lb_norm.withColumn("InLabourForce", df_lb_norm["InLabourForce"].cast(IntegerType()))

    df_lb = df_toronto.withColumn("PostCode", functions.substring_index(col("PostalCode"), " ", 1)) \
        .join(df_lb_norm, on='PostCode', how='left')

    df_lb_pandas = df_lb.toPandas()
    df_lb_pandas = df_lb_pandas.dropna() #neighbourhoods not present in the toronto data dropped
    df_lb_pandas['ratio_emply'] = df_lb_pandas['InLabourForce'] / df_lb_pandas['LabourForceCategory']
    df_lb_pandas_emply = df_lb_pandas[['BusinessID', 'Neighbourhood', 'BusinessStars', 'ratio_emply']]

    df_lb_pandas_emply = df_lb_pandas_emply.astype({'BusinessStars': 'double'})
    df_group_lb = df_lb_pandas_emply.groupby('Neighbourhood').mean()

    df_group_lb.to_csv('df_lb_pandas_emply')

if __name__ == '__main__':
    business_data = sys.argv[1]
    labour_data = sys.argv[2]
    postcode_data = sys.argv[3]
    main(business_data, labour_data, postcode_data)
