import sys
from numpy.random import randint
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import FeatureHasher
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

if len(sys.argv) != 2:
    raise Exception('Exactly 1 arguments are required: <input_file_path>')

input_file_path = sys.argv[1]

# DataFrameの作成
spark = SparkSession.builder.appName('criteo').config('spark.some.config.option', 'some-value').getOrCreate()
df = spark.read.csv(input_file_path, encoding='utf-8', inferSchema=True, sep='\t')

# 特徴量と正解ラベルの定義
label_idx = 0
real_features = [f'_c{i}' for i in range(1, 14)]
category_features = [f'_c{i}' for i in range(14, 40)]

# 欠損値の補完
df = df.fillna({feature:-1 for feature in real_features})
df = df.fillna({feature:'NULL' for feature in category_features})

# 学習・テストデータに分割
train, test = df.randomSplit([0.8, 0.2], seed=42)

# ダウンサンプリング
ratio = 1.0
counts = train.select(f'_c{label_idx}').groupBy(f'_c{label_idx}').count().collect()
higher_bound = counts[1][1]
treshold = int(ratio * float(counts[0][1]) / counts[1][1] * higher_bound)

rand_gen = lambda x: randint(0, higher_bound) if x == 0 else -1
udf_rand_gen = udf(rand_gen, IntegerType())
train = train.withColumn('rand_idx', udf_rand_gen('_c0'))
train_subsample = train.filter(train['rand_idx'] < treshold)
train_subsample = train_subsample.drop('rand_idx')

train_subsample.select(f'_c{label_idx}').groupBy(f'_c{label_idx}').count().show(n=5)

# パイプラインの構築
hasher = FeatureHasher(numFeatures=262144, inputCols=real_features + category_features, outputCol='features', categoricalCols=category_features)
lr = LogisticRegression(featuresCol='features', labelCol=f'_c{label_idx}')
pipeline = Pipeline(stages=[hasher, lr])    

model = pipeline.fit(train_subsample)
print(model.stages[-1].coefficients)

predictions = model.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol=f'_c{label_idx}', metricName='f1')
f1 = evaluator.evaluate(predictions)
print(f'f1 = {f1}')