import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('Topic distribution').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

#code takes low_topic_dist and high_topic_dist csv files as input and create csv file for topics rowwise for each buisness for visualization

#function for formating topic string
def join_text(x):
    x1 = x.replace("[", "")
    x2 = x1.replace("]", "")
    return x2
join_udf = functions.udf(lambda x: join_text(x), types.StringType())


def main(lowtopic_dist,hightopic_dist,output_folder):

    lowtopic_df = spark.read.csv(lowtopic_dist,header=True)
    hightopic_df = spark.read.csv(hightopic_dist,header=True)
    


    df_init_low = lowtopic_df.select(lowtopic_df['BusinessID'], lowtopic_df['BusinessName'], join_udf(lowtopic_df['topic_distribution']).alias('topic_distribution'))
    df_expl_low = df_init_low.select(df_init_low['BusinessID'], df_init_low['BusinessName'], \
                         functions.posexplode(functions.split(df_init_low['topic_distribution'], ','))) \
                            .withColumnRenamed('pos', 'TopicID') \
                            .withColumnRenamed('col', 'TopicDistribution')

    df_expl_low.coalesce(1).write.csv(output_folder + '/Topic_low_topic_exploded',header=True)   


    df_init_high = hightopic_df.select(hightopic_df['BusinessID'], hightopic_df['BusinessName'], join_udf(hightopic_df['topic_distribution']).alias('topic_distribution'))
    df_expl_high = df_init_high.select(df_init_high['BusinessID'], df_init_high['BusinessName'], \
                         functions.posexplode(functions.split(df_init_high['topic_distribution'], ','))) \
                            .withColumnRenamed('pos', 'TopicID') \
                            .withColumnRenamed('col', 'TopicDistribution')

    df_expl_high.coalesce(1).write.csv(output_folder + '/Topic_high_topic_exploded',header=True)  

if __name__ == '__main__':
    lowtopic_dist=sys.argv[1]
    hightopic_dist=sys.argv[2]
    output_folder=sys.argv[3]       
    main(lowtopic_dist,hightopic_dist,output_folder) 


