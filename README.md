# CS643 Programming Assignment 2 — Wine Quality Prediction on AWS

Distributed ML pipeline built on AWS EC2. Trains a wine quality classifier across 4 nodes using Apache Spark MLlib, saves the model to S3, and runs predictions on a single machine. The prediction app is also packaged as a Docker container.

**Docker Hub:** `kenil1701/wine-prediction:latest`

---

## What's in this repo

```
cs643-wine-quality-ml/
├── src/                        # Training app (WineTraining.java)
├── wine-prediction/            # Prediction app (WinePrediction.java)
│   └── src/
├── pom.xml                     # Training app Maven config
├── wine-prediction/pom.xml     # Prediction app Maven config
└── img/                        # Screenshots
```

---

## How it works

Training runs on a 4-node Spark cluster (1 master + 3 workers) on EC2. The app reads `TrainingDataset.csv` from S3, trains a Random Forest classifier, and saves the model back to S3. Prediction loads that model and scores any wine CSV, printing the F1 score.

---

## Setup

### 1. AWS — Launch 4 EC2 Instances

- AMI: Ubuntu 22.04 LTS
- Type: t2.medium
- Open ports: 22, 7077, 8080, 4040 inbound; all traffic outbound
- Note the private IPs of all 4 instances

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
# add your 3 worker private IPs, one per line:
# 172.31.36.238
# 172.31.43.212
# 172.31.36.36
```

On **all 4 machines** — set spark-env.sh:
```bash
cp /opt/spark/conf/spark-env.sh.template /opt/spark/conf/spark-env.sh
nano /opt/spark/conf/spark-env.sh
```

Add at the bottom:
```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export SPARK_MASTER_HOST=172.31.39.10
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
/opt/spark/sbin/start-worker.sh spark://172.31.39.10:7077
ssh ubuntu@172.31.36.238 "/opt/spark/sbin/start-worker.sh spark://172.31.39.10:7077"
ssh ubuntu@172.31.43.212 "/opt/spark/sbin/start-worker.sh spark://172.31.39.10:7077"
ssh ubuntu@172.31.36.36 "/opt/spark/sbin/start-worker.sh spark://172.31.39.10:7077"
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

scp /home/ubuntu/TrainingDataset.csv ubuntu@172.31.36.238:/home/ubuntu/
scp /home/ubuntu/TrainingDataset.csv ubuntu@172.31.43.212:/home/ubuntu/
scp /home/ubuntu/TrainingDataset.csv ubuntu@172.31.36.36:/home/ubuntu/
scp /home/ubuntu/ValidationDataset.csv ubuntu@172.31.36.238:/home/ubuntu/
scp /home/ubuntu/ValidationDataset.csv ubuntu@172.31.43.212:/home/ubuntu/
scp /home/ubuntu/ValidationDataset.csv ubuntu@172.31.36.36:/home/ubuntu/
```

Submit the training job:
```bash
/opt/spark/bin/spark-submit \
  --master spark://172.31.39.10:7077 \
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
=========================================
🍷 Wine Quality Prediction F1 Score: 0.5260
=========================================
```

---

## Running the Prediction App with Docker

Pull and run the container:
```bash
docker pull kenil1701/wine-prediction:latest

docker run --rm \
  -e AWS_ACCESS_KEY_ID=<your_key> \
  -e AWS_SECRET_ACCESS_KEY=<your_secret> \
  -e AWS_SESSION_TOKEN=<your_token> \
  kenil1701/wine-prediction:latest
```

---

## Model

Random Forest Classifier via Spark MLlib. 100 trees, trained on 11 chemical features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol). Predicts wine quality on a scale of 1–10.

Validation F1 score: **0.5260**

---

## Links

- **GitHub:** https://github.com/kenilavaiyaa/cs643-wine-quality-ml
- **Docker Hub:** https://hub.docker.com/r/kenil1701/wine-prediction
