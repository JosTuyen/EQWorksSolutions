from pyspark.sql import SparkSession
from pyspark.sql import Row
from graphframes import *
import math

def calDist(source,targets):
    nearestP = ''
    dis = 0
    for target in targets:
        tmpDist = (float(source['Latitude']) - float(target[' Latitude']))**2 + (float(source['Longitude']) - float(target['Longitude']))**2
        if (dis > tmpDist or dis == 0):
            dis = tmpDist
            nearestP = target['POIID']
    return Row(**source.asDict(), label=nearestP, distance=dis)

def main():
    spark = SparkSession.builder.master('local').appName("EQWork").getOrCreate()
    POIdf = spark.read.format("csv").option("header", "true").load("/tmp/data/POIList.csv")
    sampleDf = spark.read.format("csv").option("header", "true").load("/tmp/data/DataSample.csv")
    cleanSample = sampleDf.dropDuplicates([' TimeSt','Latitude','Longitude'])
    poiDf = POIdf.drop_duplicates([' Latitude','Longitude'])
    poiList = poiDf.rdd.collect()
    labelSample = cleanSample.rdd.map(lambda r: calDist(r,poiList)).toDF()
    avg = labelSample.groupBy('label').agg({'distance':'avg'})
    stddev = labelSample.groupBy('label').agg({'distance':'stddev'})
    radius = labelSample.groupBy('label').agg({'distance':'max'}).orderBy('label').collect()
    total = labelSample.groupBy('label').count().orderBy('label').collect()
    dense = [total[x]['count']/(math.pi * radius[x]['max(distance)']**2) for x in range(3)]
    print(dense)
    verticesList = open('/tmp/data/task_ids.txt').read().split(',')
    rowList = [Row(id = x) for x in verticesList]
    rowDf = spark.createDataFrame(rowList)
    relations = spark.read.text('/tmp/data/relations.txt').rdd.map(lambda l : Row(src=l[0].split('->')[1],dst=l[0].split('->')[0])).toDF()
    g = GraphFrame(rowDf,relations)
    g.vertices.show()
    g.edges.show()
    g.degrees.show()
if __name__ == '__main__':
    main()
