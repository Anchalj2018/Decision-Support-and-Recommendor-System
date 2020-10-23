import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+


from pyspark.sql import SparkSession, functions, types
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col


spark = SparkSession.builder.appName('Business').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext


# main function
def main(sentiment_input,user_input,review_input,output_folder):
    # read input files
    df_sentiment = spark.read.csv(sentiment_input, header=True)
    df_user = spark.read.parquet(user_input)
    df_review = spark.read.parquet(review_input)

    # join review and users
    df_rev_usr = df_review.join(df_user, df_review['UserID'] == df_user['UserID'], 'inner') \
                        .select(df_review['BusinessID'], \
                                df_user['UserID'], \
                                df_user['UserName'], \
                                df_user['ReviewCount'].alias('UserReviewCount'), \
                                df_user['AverageStars'].alias('UserAverageStars'), \
                                df_review['ReviewStars'], \
                                functions.dayofyear(df_review['Date']).alias('ReviewDayOfYear'))

    # join with business sentiments
    df_rev_usr_bus = df_sentiment.join(df_rev_usr, df_rev_usr['BusinessID'] == df_sentiment['BusinessID'], 'inner') \
                             .where(df_sentiment['ZipCode'].isNull() == False) \
                             .select(df_rev_usr['*'], \
                                    df_sentiment['Name'].alias('BusinessName'), \
                                    df_sentiment['ZipCode'].alias('BusinessPostalCode'), \
                                    df_sentiment['ZipCode'].substr(1, 3).alias('BusinessNeighborhood'), \
                                    df_sentiment['Latitude'].cast(types.FloatType()), \
                                    df_sentiment['Longitude'].cast(types.FloatType()), \
                                    df_sentiment['avg_neg'].cast(types.FloatType()).alias('AverageNegative'), \
                                    df_sentiment['avg_neu'].cast(types.FloatType()).alias('AverageNeutral'), \
                                    df_sentiment['avg_pos'].cast(types.FloatType()).alias('AveragePositive'), \
                                    df_sentiment['avg_composite_score'].cast(types.FloatType()).alias('AverageComposite'))

    # prepare train and validation set
    train, validation = df_rev_usr_bus.randomSplit([0.75, 0.25])
    train = train.cache()
    validation = validation.cache()

    # convert neighborhood to indexed
    indexer = StringIndexer(inputCol="BusinessNeighborhood", outputCol="BusinessNeighborhoodIndexed")

    # input columns
    assembler = VectorAssembler(inputCols=['UserReviewCount', 'UserAverageStars', 'Latitude', 'Longitude', 'BusinessNeighborhoodIndexed', \
                                           'ReviewDayOfYear', 'AverageNegative', 'AverageNeutral', 'AveragePositive', 'AverageComposite'], \
                                outputCol='features')

    # output column
    regressor = GBTRegressor(featuresCol='features', labelCol='ReviewStars', maxIter=5, maxDepth=10, maxBins=110)

    # pipeline
    pipeline = Pipeline(stages=[indexer, assembler, regressor])

    # train model
    model = pipeline.fit(train)

    # make predictions
    predictions = model.transform(validation)

    # evaluate model
    r2_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='ReviewStars', metricName='r2')
    r2 = r2_evaluator.evaluate(predictions)

    rmse_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='ReviewStars', metricName='rmse')
    rmse = rmse_evaluator.evaluate(predictions)

    print("r2: %f" % (r2))
    print("rmse: %f" % (rmse))

    model.write().overwrite().save(output_folder + '/TrainedModel')

if __name__ == '__main__':
    sentiment_input=sys.argv[1]
    user_input=sys.argv[2]
    review_input = sys.argv[3]
    output_folder =sys.argv[4]
    main(sentiment_input,user_input,review_input,output_folder)
