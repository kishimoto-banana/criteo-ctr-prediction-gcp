import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

if len(sys.argv) != 2:
    raise Exception('Exactly 1 arguments are required: <input_file_path>')

input_file_path = sys.argv[1]

# DataFrameの作成
from pyspark.sql import SparkSession
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

# パイプラインの構築
indexers = [StringIndexer(inputCol=feature, outputCol=f'{feature}_idx',  handleInvalid='keep') for feature in category_features]
encoder = OneHotEncoderEstimator(inputCols=[f'{feature}_idx' for feature in category_features], outputCols=[f'{feature}_vec' for feature in category_features], dropLast=False, handleInvalid='keep')
assembler = VectorAssembler(inputCols=real_features+[f'{feature}_vec' for feature in category_features], outputCol='assembles')
pca = PCA(k=100, inputCol='assembles', outputCol='features')
lr = LogisticRegression(featuresCol='features', labelCol=f'_c{label_idx}', maxIter=100)
stages = indexers
stages.append(encoder)
stages.append(assembler)
stages.append(pca)
stages.append(lr)
pipeline = Pipeline(stages=stages)

model = pipeline.fit(train)
print(model.stages[-1].coefficients)

predictions = model.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol=f'_c{label_idx}', metricName='f1')
f1 = evaluator.evaluate(predictions)
print(f'f1 = {f1}')