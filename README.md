# Wine Quality Prediction on AWS

Distributed ML pipeline built on AWS EC2. Trains a wine quality classifier across 4 nodes using Apache Spark MLlib, saves the model to S3, and runs predictions on a single machine. The prediction app is also packaged as a Docker container.

**Docker Hub:** `https://hub.docker.com/repository/docker/kenil1701/wine-prediction/general`

---

## What's in this repo

```
wine-project/
├── src/                        # Training app (WineTraining.java)
├── wine-prediction/            # Prediction app (WinePrediction.java)
│   └── src/
├── pom.xml                     # Training app Maven config
└── wine-prediction/pom.xml     # Prediction app Maven config
```

---

## How it works

Training runs on a four-node Spark cluster (one master node and three worker nodes) on Amazon EC2. The application reads the `TrainingDataset.csv` file from Amazon S3, trains a Random Forest classifier, and saves the trained model back to the same S3 location. Subsequently, the prediction module loads the saved model and performs predictions on any wine CSV file. It then prints the F1 score for each prediction.

---

## Setup

### 1. AWS — Launch 4 EC2 Instances

Upload your datasets to S3:
```bash
aws s3 cp TrainingDataset.csv s3://kenil-cs-wine/
aws s3 cp ValidationDataset.csv s3://kenil-cs-wine/
```

### 2. Install Java and Spark (all 4 machines)

```bash
sudo apt update
sudo apt install -y openjdk-11-jdk

wget https://archive.apache.org/dist/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz
tar -xvzf spark-3.3.2-bin-hadoop3.tgz
sudo mv spark-3.3.2-bin-hadoop3 /opt/spark

echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bashrc
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc
```

Install AWS CLI:
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install -y unzip
unzip awscliv2.zip
sudo ./aws/install
```

Configure credentials (repeat on all 4 machines — credentials expire each AWS Academy session):
```bash
mkdir -p ~/.aws
nano ~/.aws/credentials
# paste your [default] key, secret, and session_token from AWS Academy > AWS Details
aws configure set region us-east-1
```

### 3. Configure the Spark Cluster

On **master only** — add worker IPs:
```bash
nano /opt/spark/conf/workers
# add your 3 worker private IPs
```

On **all 4 machines** — set spark-env.sh:
```bash
cp /opt/spark/conf/spark-env.sh.template /opt/spark/conf/spark-env.sh
nano /opt/spark/conf/spark-env.sh
```

Add at the bottom:
```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export SPARK_MASTER_HOST=<master private ip>
export SPARK_LOCAL_IP=$(hostname -I | tr -d ' ')
```

Set up passwordless SSH from master to workers:
```bash
# on master
ssh-keygen -t rsa -P "" -f ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub
# copy output, then on each worker:
nano ~/.ssh/authorized_keys  # paste master's public key
chmod 600 ~/.ssh/authorized_keys
```

Start the cluster:
```bash
/opt/spark/sbin/start-master.sh
/opt/spark/sbin/start-worker.sh spark://<master private ip>:7077
ssh ubuntu@<slave1 private ip> "/opt/spark/sbin/start-worker.sh spark://<master private ip>:7077"
ssh ubuntu@<slave2 private ip> "/opt/spark/sbin/start-worker.sh spark://<master private ip>:7077"
ssh ubuntu@<slave3 private ip> "/opt/spark/sbin/start-worker.sh spark://<master private ip>:7077"
```

Check it at `http://<MASTER_PUBLIC_IP>:8080` — you should see 3 alive workers.

### 4. Download S3 JARs (master only)

```bash
wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.2/hadoop-aws-3.3.2.jar
wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar
```

---

## Build

Clone this repo on your local machine and build both JARs with Maven:

```bash
# Training app
mvn clean package
# JAR: target/wine-training-1.0-SNAPSHOT.jar

# Prediction app
cd wine-prediction
mvn clean package
# JAR: target/wine-prediction-1.0-SNAPSHOT.jar
```

Upload both to master:
```bash
scp -i your-key.pem target/wine-training-1.0-SNAPSHOT.jar ubuntu@<MASTER_PUBLIC_IP>:/home/ubuntu/
scp -i your-key.pem wine-prediction/target/wine-prediction-1.0-SNAPSHOT.jar ubuntu@<MASTER_PUBLIC_IP>:/home/ubuntu/
```

---

## Running the Training App (4 EC2 nodes)

On master, copy datasets locally and to all workers:
```bash
aws s3 cp s3://kenil-cs-wine/TrainingDataset.csv /home/ubuntu/
aws s3 cp s3://kenil-cs-wine/ValidationDataset.csv /home/ubuntu/

scp /home/ubuntu/TrainingDataset.csv ubuntu@<slave1 private ip>:/home/ubuntu/
scp /home/ubuntu/TrainingDataset.csv ubuntu@<slave2 private ip>:/home/ubuntu/
scp /home/ubuntu/TrainingDataset.csv ubuntu@<slave3 private ip>:/home/ubuntu/
scp /home/ubuntu/ValidationDataset.csv ubuntu@<slave1 private ip>:/home/ubuntu/
scp /home/ubuntu/ValidationDataset.csv ubuntu@<slave2 private ip>:/home/ubuntu/
scp /home/ubuntu/ValidationDataset.csv ubuntu@<slave3 private ip>:/home/ubuntu/
```

Submit the training job:
```bash
/opt/spark/bin/spark-submit \
  --master spark://<master ip address>:7077 \
  --deploy-mode client \
  --num-executors 3 \
  --executor-memory 1g \
  --executor-cores 1 \
  --driver-memory 2g \
  --jars /home/ubuntu/hadoop-aws-3.3.2.jar,/home/ubuntu/aws-java-sdk-bundle-1.11.1026.jar \
  --conf spark.hadoop.fs.s3a.access.key=$(aws configure get aws_access_key_id) \
  --conf spark.hadoop.fs.s3a.secret.key=$(aws configure get aws_secret_access_key) \
  --conf spark.hadoop.fs.s3a.session.token=$(aws configure get aws_session_token) \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider \
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
  --conf spark.driver.host=172.31.39.10 \
  --class com.wine.WineTraining \
  /home/ubuntu/wine-training-1.0-SNAPSHOT.jar
```

The model saves to `s3://kenil-cs-wine/wine-model` when done.

---

## Running the Prediction App (single EC2 node, no Docker)

```bash
/opt/spark/bin/spark-submit \
  --master local[*] \
  --driver-memory 2g \
  --jars /home/ubuntu/hadoop-aws-3.3.2.jar,/home/ubuntu/aws-java-sdk-bundle-1.11.1026.jar \
  --conf spark.hadoop.fs.s3a.access.key=$(aws configure get aws_access_key_id) \
  --conf spark.hadoop.fs.s3a.secret.key=$(aws configure get aws_secret_access_key) \
  --conf spark.hadoop.fs.s3a.session.token=$(aws configure get aws_session_token) \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider \
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
  --class com.wine.WinePrediction \
  /home/ubuntu/wine-prediction-1.0-SNAPSHOT.jar \
  /home/ubuntu/ValidationDataset.csv \
  s3a://kenil-cs-wine/wine-model
```

Expected output:
```
====================================================
Wine Quality Prediction F1 Score: 0.5259500915750914
====================================================
```

---

## Docker Implementation

### Dockerfile

```dockerfile
FROM eclipse-temurin:11-jre-jammy

RUN apt-get update && apt-get install -y wget && \
    wget -q https://archive.apache.org/dist/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz && \
    tar -xzf spark-3.3.2-bin-hadoop3.tgz && \
    mv spark-3.3.2-bin-hadoop3 /opt/spark && \
    rm spark-3.3.2-bin-hadoop3.tgz

COPY wine-prediction-1.0-SNAPSHOT.jar /app/
COPY hadoop-aws-3.3.2.jar /app/
COPY aws-java-sdk-bundle-1.11.1026.jar /app/
COPY ValidationDataset.csv /app/

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

ENTRYPOINT ["/opt/spark/bin/spark-submit", \
    "--master", "local[*]", \
    "--jars", "/app/hadoop-aws-3.3.2.jar,/app/aws-java-sdk-bundle-1.11.1026.jar", \
    "--class", "com.wine.WinePrediction", \
    "/app/wine-prediction-1.0-SNAPSHOT.jar", \
    "/app/ValidationDataset.csv"]
```

### Install Docker on EC2

```bash
sudo apt-get update && sudo apt-get install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker ubuntu
newgrp docker
```

### Build the Docker Image (on EC2 master)

Make sure all required files are in `/home/ubuntu/`:
```bash
ls /home/ubuntu/wine-prediction-1.0-SNAPSHOT.jar
ls /home/ubuntu/hadoop-aws-3.3.2.jar
ls /home/ubuntu/aws-java-sdk-bundle-1.11.1026.jar
ls /home/ubuntu/ValidationDataset.csv
ls /home/ubuntu/Dockerfile
```

Build the image:
```bash
cd /home/ubuntu
docker build -t wine-prediction .
```

### Test the Docker Container Locally

```bash
docker run --rm wine-prediction
```

### Push to Docker Hub

Login to Docker Hub:
```bash
docker login
```

Tag the image:
```bash
docker tag wine-prediction kenil1701/wine-prediction:latest
```

Push to Docker Hub:
```bash
docker push kenil1701/wine-prediction:latest
```

### Pull and Run from Docker Hub (on any machine)

```bash
docker pull kenil1701/wine-prediction:latest

docker run --rm kenil1701/wine-prediction:latest
```
---

## Model

Random Forest Classifier implemented using Spark MLlib. The model comprises 100 trees and is trained on 11 chemical features, including fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulfates, and alcohol. The classifier predicts wine quality on a scale of 1–10.

Validation F1 score: **0.5259500915750914**

---

## Links

- **GitHub:** https://github.com/KenilAvaiyaa/cloud-project-wine-quality/tree/main
- **Docker Hub:** https://hub.docker.com/repository/docker/kenil1701/wine-prediction/general
