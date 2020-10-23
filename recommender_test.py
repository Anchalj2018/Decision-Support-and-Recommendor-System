import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+


from pyspark.sql import SparkSession, functions, types
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
from pyspark.ml import PipelineModel


spark = SparkSession.builder.appName('Business').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext


# main function
def main(sentiment_input,user_input,review_input,model_input,output_folder):
    # read input files
    df_sentiment = spark.read.csv(sentiment_input, header=True)
    df_user = spark.read.parquet(user_input)
    df_review = spark.read.parquet(review_input)

    # get 50 users
    df_50_users = df_user.limit(50)

    # cross join user and business
    df_usr_bus_all = df_50_users \
                    .crossJoin(df_sentiment) \
                    .where(df_sentiment['ZipCode'].isNull() == False) \
                            .select(
                                df_sentiment['BusinessID'], \
                                df_user['UserID'], \
                                df_user['UserName'], \
                                df_user['ReviewCount'].alias('UserReviewCount'), \
                                df_user['AverageStars'].alias('UserAverageStars'), \
                                functions.lit(0).alias('ReviewStars'), \
                                functions.dayofyear(functions.current_date()).alias('ReviewDayOfYear'), \
                                df_sentiment['Name'].alias('BusinessName'), \
                                df_sentiment['ZipCode'].alias('BusinessPostalCode'), \
                                df_sentiment['ZipCode'].substr(1, 3).alias('BusinessNeighborhood'), \
                                df_sentiment['Latitude'].cast(types.FloatType()), \
                                df_sentiment['Longitude'].cast(types.FloatType()), \
                                df_sentiment['avg_neg'].cast(types.FloatType()).alias('AverageNegative'), \
                                df_sentiment['avg_neu'].cast(types.FloatType()).alias('AverageNeutral'), \
                                df_sentiment['avg_pos'].cast(types.FloatType()).alias('AveragePositive'), \
                                df_sentiment['avg_composite_score'].cast(types.FloatType()).alias('AverageComposite'))

    # left join with reviews
    df_joined = df_usr_bus_all.join(df_review, ['BusinessID', 'UserID'], 'left_outer') \
                            .select(df_review['ReviewID'], \
                                    df_usr_bus_all['BusinessID'], \
                                    df_usr_bus_all['UserID'], \
                                    df_usr_bus_all['UserName'], \
                                    df_usr_bus_all['UserReviewCount'], \
                                    df_usr_bus_all['UserAverageStars'], \
                                    df_usr_bus_all['ReviewStars'], \
                                    df_usr_bus_all['ReviewDayOfYear'], \
                                    df_usr_bus_all['BusinessName'], \
                                    df_usr_bus_all['BusinessPostalCode'], \
                                    df_usr_bus_all['BusinessNeighborhood'], \
                                    df_usr_bus_all['Latitude'], \
                                    df_usr_bus_all['Longitude'], \
                                    df_usr_bus_all['AverageNegative'], \
                                    df_usr_bus_all['AverageNeutral'], \
                                    df_usr_bus_all['AveragePositive'], \
                                    df_usr_bus_all['AverageComposite'])

    # get restaurants that user has not visited
    df_not_visited_rests = df_joined.where(df_joined['ReviewID'].isNull())

    # load the model
    loaded_model = PipelineModel.load(model_input)

    # use the model to make predictions
    predictions = loaded_model.transform(df_not_visited_rests)
    predictions_init = predictions.select(predictions['BusinessID'], \
                                          predictions['BusinessName'], \
                                          predictions['BusinessPostalCode'], \
                                          predictions['BusinessNeighborhood'], \
                                          predictions['UserID'], \
                                          predictions['UserName'], \
                                          predictions['UserReviewCount'], \
                                          predictions['UserAverageStars'], \
                                          predictions['ReviewDayOfYear'], \
                                          predictions['prediction'].alias('PredictedReviewStar'), \
                                          predictions['Latitude'], \
                                          predictions['Longitude'], \
                                          predictions['AverageNegative'], \
                                          predictions['AverageNeutral'], \
                                          predictions['AveragePositive'], \
                                          predictions['AverageComposite'])

    # change scores > 5 to 5 and < 0 to 0
    predictions_final = predictions_init.withColumn('FinalStar', \
                                                        functions.when(predictions_init["PredictedReviewStar"] >= 5, 5) \
                                                        .otherwise(functions.when(predictions_init["PredictedReviewStar"] <= 0, 0) \
                                                        .otherwise(predictions_init['PredictedReviewStar'])))

    # partition By user
    window = Window.partitionBy(predictions_final['UserID']).orderBy(predictions_final['FinalStar'].desc())

    # get top 10 scores for each user based on partition
    prediction_to_save = predictions_final.select('*', functions.row_number().over(window).alias('rank')).filter(col('rank') <= 10)

    # save predictions to output
    prediction_to_save.coalesce(1).write.csv(output_folder + '/TestModel', header=True)

if __name__ == '__main__':
    sentiment_input=sys.argv[1]
    user_input=sys.argv[2]
    review_input = sys.argv[3]
    model_input =sys.argv[4]
    output_folder=sys.argv[5]
    main(sentiment_input,user_input,review_input,model_input,output_folder)
