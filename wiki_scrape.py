import pyspark
import pandas as pd
import wikipedia as wp
import sys

spark = SparkSession.builder.appName('Wikipedia Scrape').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')


def main():
    html = wp.page("List of postal codes of Canada: M").html().encode("UTF-8")
    df_postcode = pd.read_html(html, header = 0)[0]
    df_postcode.columns = df_postcode.columns.str.strip()
    df_postcode = spark.createDataFrame(df_postcode)
    
    df_postcode = df_postcode.filter(df_postcode['Borough'] != 'Not assigned')
    df_postcode = df_postcode.filter(df_postcode['Neighbourhood'] != 'Not assigned')
    df_postcode.coalesce(1).write.csv('postcode_wiki', sep = ',', header=True)
    
if __name__ == '__main__':
    main()


