FROM apache/spark:3.5.0

USER root

RUN apt-get update && apt-get install -y python3-pip wget
RUN pip install kafka-python psycopg2-binary pandas

# PostgreSQL JDBC 
RUN wget -P /opt/spark/jars https://jdbc.postgresql.org/download/postgresql-42.7.3.jar

# spark-kafka connector and kafka-clients 
RUN wget -P /opt/spark/jars \
  https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.0/spark-sql-kafka-0-10_2.12-3.5.0.jar && \
  wget -P /opt/spark/jars \
  https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.0/spark-token-provider-kafka-0-10_2.12-3.5.0.jar && \
  wget -P /opt/spark/jars \
  https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.6.0/kafka-clients-3.6.0.jar

# commons-pool2
RUN wget -P /opt/spark/jars https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar

COPY stream_processor.py /opt/spark/app/
COPY load_historical.py /opt/spark/app/
COPY tunisia_5y.json /opt/spark/app/

WORKDIR /opt/spark/app/

CMD ["sleep", "infinity"]
