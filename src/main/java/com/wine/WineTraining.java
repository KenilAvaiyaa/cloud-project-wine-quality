package com.wine;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class WineTraining {

    
    public static void main(String[] args) {

        
        SparkSession spark = SparkSession.builder()
                .appName("WineQualityTraining")
                .getOrCreate();

   

        System.out.println("✅ Spark session started successfully.");

       
        Dataset<Row> trainingData = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("sep", ";")
                .csv("/home/ubuntu/TrainingDataset.csv");



        
        Dataset<Row> validationData = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("sep", ";")
                .csv("/home/ubuntu/ValidationDataset.csv");

        // Clean column names - strip all extra quotes and whitespace
        for (String col : trainingData.columns()) {
            String cleanCol = col.replaceAll("\"", "").trim();
            trainingData = trainingData.withColumnRenamed(col, cleanCol);
        }
        for (String col : validationData.columns()) {
            String cleanCol = col.replaceAll("\"", "").trim();
            validationData = validationData.withColumnRenamed(col, cleanCol);
        }

        System.out.println("✅ Data loaded and columns cleaned from S3");
        System.out.println("Training rows: " + trainingData.count());
        System.out.println("Validation rows: " + validationData.count());
        System.out.println("Columns: " + java.util.Arrays.toString(trainingData.columns()));

       
        String[] featureColumns = {
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol"
        };

       
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureColumns)
                .setOutputCol("features");

       
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("quality")     // column we want to predict
                .setFeaturesCol("features") // the vector column created above
                .setNumTrees(100)           // more trees = better accuracy (up to a point)
                .setSeed(42);              // fixed seed for reproducibility

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{assembler, rf});

        System.out.println("🚀 Training model across the cluster — this may take a few minutes...");
        PipelineModel model = pipeline.fit(trainingData);
        System.out.println("✅ Model trained successfully.");

        Dataset<Row> predictions = model.transform(validationData);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1Score = evaluator.evaluate(predictions);
        System.out.printf("✅ Validation F1 Score: %.4f%n", f1Score);

        try {
            // Force all computations to complete before saving
            predictions.cache();
            predictions.count();
            
            model.write().overwrite().save("s3a://kenil-cs-wine/wine-model");
            System.out.println("✅ Model saved to s3a://kenil-cs-wine/wine-model");
            System.out.println("✅ Verification - files saved: " + 
                new java.io.File("/home/ubuntu/wine-model/stages").listFiles().length + " stages");
        } catch (Exception e) {
            System.err.println("❌ Error saving model: " + e.getMessage());
            e.printStackTrace();
        }

        // For stop the Spark session and release cluster resources
        spark.stop();
    }
}