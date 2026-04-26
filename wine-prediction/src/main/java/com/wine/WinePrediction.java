package com.wine;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WinePrediction {
    public static void main(String[] args) {

        if (args.length < 1) {
            System.err.println("❌ Usage: WinePrediction <path-to-test-csv>");
            System.exit(1);
        }

        String testDataPath = args[0];
        String modelPath = "s3a://kenil-cs-wine/wine-model";

        System.out.println("🍷 Wine Quality Prediction App");
        System.out.println("📂 Test data: " + testDataPath);
        System.out.println("📦 Model path: " + modelPath);

        SparkSession spark = SparkSession.builder()
                .appName("WineQualityPrediction")
                .getOrCreate();

        System.out.println("✅ Spark session started");

        Dataset<Row> testData = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("sep", ";")
                .csv(testDataPath);

        for (String col : testData.columns()) {
            String cleanCol = col.replaceAll("\"", "").trim();
            testData = testData.withColumnRenamed(col, cleanCol);
        }

        System.out.println("✅ Test data loaded: " + testData.count() + " rows");
        System.out.println("Columns: " + 
            java.util.Arrays.toString(testData.columns()));

        PipelineModel model = PipelineModel.load(modelPath);
        System.out.println("✅ Model loaded from " + modelPath);

        Dataset<Row> predictions = model.transform(testData);
        System.out.println("✅ Predictions complete");

        MulticlassClassificationEvaluator evaluator =
                new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1Score = evaluator.evaluate(predictions);

        System.out.println("=========================================");
        System.out.println("🍷 Wine Quality Prediction F1 Score: " + f1Score);
        System.out.println("=========================================");

        spark.stop();
    }
}