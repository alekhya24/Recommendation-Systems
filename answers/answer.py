import os
import sys
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.functions import desc
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType
from pyspark import SparkContext
sc =SparkContext()

'''
INTRODUCTION

With this assignment you will get a practical hands-on of recommender
systems in Spark. To begin, make sure you understand the example
at http://spark.apache.org/docs/latest/ml-collaborative-filtering.html
and that you can run it successfully. 

We will use the MovieLens dataset sample provided with Spark and
available in directory `data`.

'''

'''
HELPER FUNCTIONS

These functions are here to help you. Instructions will tell you when
you should use them. Don't modify them!
'''

#Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

#Useful functions to print RDDs and Dataframes.
def toCSVLineRDD(rdd):
    '''
    This function convert an RDD or a DataFrame into a CSV string
    '''
    a = rdd.map(lambda row: ",".join([str(elt) for elt in row]))\
           .reduce(lambda x,y: os.linesep.join([x,y]))
    return a + os.linesep

def toCSVLine(data):
    '''
    Convert an RDD or a DataFrame into a CSV string
    '''
    if isinstance(data, RDD):
        return toCSVLineRDD(data)
    elif isinstance(data, DataFrame):
        return toCSVLineRDD(data.rdd)
    return None

def basic_als_recommender(filename, seed):
    '''
    This function must print the RMSE of recommendations obtained
    through ALS collaborative filtering, similarly to the example at
    http://spark.apache.org/docs/latest/ml-collaborative-filtering.html
    The training ratio must be 80% and the test ratio must be 20%. The
    random seed used to sample the training and test sets (passed to
    ''DataFrame.randomSplit') is an argument of the function. The seed
    must also be used to initialize the ALS optimizer (use
    *ALS.setSeed()*). The following parameters must be used in the ALS
    optimizer:
    - maxIter: 5
    - rank: 70
    - regParam: 0.01
    - coldStartStrategy: 'drop'
    Test file: tests/test_basic_als.py
    '''
    spark=init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2])))
    ratings =spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2],seed)
    als = ALS(rank=70,maxIter=5, regParam=0.01,userCol="userId", itemCol="movieId", ratingCol="rating",coldStartStrategy="drop")
    als.setSeed(seed)
    model = als.fit(training)
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("RMSE:{0}".format(rmse))
    return rmse

def global_average(filename, seed):
    '''
    This function must print the global average rating for all users and[
    all movies in the training set. Training and test
    sets should be determined as before (e.g: as in function basic_als_recommender).
    Test file: tests/test_global_average.py
    '''
    spark=init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2])))
    ratings =spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2],seed)
    global_avg = training.agg({"rating": "mean"}).collect()[0][0]
    print("Global_avg:{0}".format(global_avg))
    return float(global_avg)

def global_average_recommender(filename, seed):
    '''
    This function must print the RMSE of recommendations obtained
    through global average, that is, the predicted rating for each
    user-movie pair must be the global average computed in the previous
    task. Training and test
    sets should be determined as before. You can add a column to an existing DataFrame with function *.withColumn(...)*.
    Test file: tests/test_global_average_recommender.py
    '''
    spark=init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2])))
    ratings =spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2],seed)
    global_avg = training.agg({"rating": "mean"}).collect()[0][0]
    test_with_global_average = test.withColumn("prediction", lit(global_avg))
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
    test_avg_RMSE = evaluator.evaluate(test_with_global_average)
    print("RMSE through Global Average:{0}".format(test_avg_RMSE))
    return test_avg_RMSE
        
def means_and_interaction(filename, seed, n):
    '''
    This function must return the *n* first elements of a DataFrame
    containing, for each (userId, movieId, rating) triple, the
    corresponding user mean (computed on the training set), item mean
    (computed on the training set) and user-item interaction *i* defined
    as *i=rating-(user_mean+item_mean-global_mean)*. *n* must be passed on
    the command line. The DataFrame must contain the following columns:

    - userId # as in the input file
    - movieId #  as in the input file
    - rating # as in the input file
    - user_mean # computed on the training set
    - item_mean # computed on the training set
    - user_item_interaction # i = rating - (user_mean+item_mean-global_mean)

    Rows must be ordered by ascending userId and then by ascending movieId.

    Training and test sets should be determined as before.
    Test file: tests/test_means_and_interaction.py

    Note, this function should return a list of collected Rows. Please, have a
    look at the test file to ensure you have the right format.
    '''
    spark=init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD=parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2])))
    ratings =spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2],seed)
    global_mean = training.agg({"rating": "mean"}).collect()[0][0]
    each_user_mean = training.groupBy("userId").agg({"rating":"mean"})
    each_item_mean = training.groupBy("movieId").agg({"rating":"mean"})
    op_df=training.orderBy("userId","movieId")
    schema=StructType([StructField('userId', IntegerType()),
                                                         StructField('movieId', IntegerType()),
                                                         StructField('rating', FloatType()),
                                                         StructField('user_mean', FloatType()),
                                                            StructField('item_mean', FloatType()),
                                                            StructField('user_item_interaction', FloatType())])
    final_df = spark.createDataFrame(sc.emptyRDD(), schema)
    sorted_training_data =op_df.take(n)
    l = []
    for i in sorted_training_data:
        user_mean = each_user_mean.filter(each_user_mean['userId']==i.userId).select('avg(rating)').collect()[0][0]
        item_mean = each_item_mean.filter(each_item_mean['movieId']==i.movieId).select('avg(rating)').collect()[0][0]
        user_item_interaction =i.rating-(user_mean+ item_mean - global_mean)
        l = l + [([i.userId,i.movieId,i.rating,user_mean,item_mean,user_item_interaction])]
    temp_df = spark.createDataFrame(l, schema)
    final_df = final_df.union(temp_df)
    return final_df.take(n);   

def als_with_bias_recommender(filename, seed):
    '''3
    This function must return the RMSE of recommendations obtained 
    using ALS + biases. Your ALS model should make predictions for *i*, 
    the user-item interaction, then you should recompute the predicted 
    rating with the formula *i+user_mean+item_mean-m* (*m* is the 
    global rating). The RMSE should compare the original rating column 
    and the predicted rating column.  Training and test sets should be 
    determined as before. Your ALS model should use the same parameters 
    as before and be initialized with the random seed passed as 
    parameter. Test file: tests/test_als_with_bias_recommender.py
    '''
    spark=init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD=parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2])))
    ratings =spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2],seed)
    global_mean = training.agg({"rating": "mean"}).collect()[0][0]
    each_user_mean = training.groupBy("userId").agg({"rating":"mean"})
    each_item_mean = training.groupBy("movieId").agg({"rating":"mean"})
    sorted_training_data=training.orderBy("userId","movieId")
    schema=StructType([StructField('userId', IntegerType()),
                                                         StructField('movieId', IntegerType()),
                                                         StructField('rating', FloatType()),
                                                         StructField('user_mean', FloatType()),
                                                        StructField('item_mean', FloatType()),
                                                        StructField('user_item_interaction', FloatType())])
    final_df = spark.createDataFrame(sc.emptyRDD(), schema)
    l = []
    for i in test.collect():
        user_mean = each_user_mean.filter(each_user_mean['userId']==i.userId).select('avg(rating)').collect()[0][0]
        item_mean = each_item_mean.filter(each_item_mean['movieId']==i.movieId).select('avg(rating)').collect()[0][0]
        user_item_interaction =i.rating-(user_mean+ item_mean - global_mean)
        l = l + [([i.userId,i.movieId,i.rating,user_mean,item_mean,user_item_interaction])]
    temp_df = spark.createDataFrame(l, schema)
    final_df = final_df.union(temp_df)
    '''(final_training,final_test) = final_df.randomSplit(0.8,0.2)'''
    als= ALS(rank=70,maxIter=5, regParam=0.01,userCol="userId", itemCol="movieId", ratingCol="rating",coldStartStrategy="drop")
    als.setSeed(seed)
    model = als.fit(final_df)
    predict_df = model.transform(test)
    data = predict_df.join(final_df).map(lambda tup: tup[1])
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
    rmse = evaluator.evaluate(data)
    print("RMSE:{0}".format(rmse))
    return rmse
