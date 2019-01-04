import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
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
df.show()

# 特徴量と正解ラベルの定義
label_idx = 0
real_features = [f'_c{i}' for i in range(1, 14)]
category_features = [f'_c{i}' for i in range(14, 40)]

# 欠損値の補完
df = df.fillna({feature:-1 for feature in real_features})
df = df.fillna({feature:'NULL' for feature in category_features})

# 文字列のラベル化
# indexers = [StringIndexer(inputCol=feature, outputCol=f'{feature}_idx').fit(df) for feature in category_features]
# pipeline = Pipeline(stages=indexers)
# indexed = pipeline.fit(df).transform(df)

indexed = df
for feature in category_features:
    indexer = StringIndexer(inputCol=feature, outputCol=f'{feature}_idx')
    model = indexer.fit(indexed)
    indexed = model.transform(indexed)

# one-hot encoding
indexed_features = [f'{feature}_idx' for feature in category_features]
onehot_features = [f'{feature}_vec' for feature in category_features]
encoded = indexed
for indexed_feature, onehot_feature in zip(indexed_features, onehot_features):
    encoder = OneHotEncoder(inputCol=indexed_feature, outputCol=onehot_feature, dropLast=False)
    encoded = encoder.transform(encoded)

# モデルの入力データへの変換
assembler = VectorAssembler(inputCols=real_features+onehot_features, outputCol='assembles')
assembled = assembler.transform(encoded)

# PCA
pca = PCA(k=2, inputCol='assembles', outputCol='features')
model = pca.fit(assembled)
X = model.transform(assembled)

# 学習とテストに分割
train, test = X.randomSplit([0.8, 0.2], seed=42)
train.show()

# ロジスティック回帰の学習
lr = LogisticRegression(featuresCol='features', labelCol=f'_c{label_idx}', maxIter=100)
model = lr.fit(train)
print('Coefficients: ' + str(model.coefficients))
print('Intercept: ' + str(model.intercept))

# 予測
predictions = model.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol=f'_c{label_idx}', metricName='f1')
f1 = evaluator.evaluate(predictions)
print(f'f1 = {f1}')