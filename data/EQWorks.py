from pyspark.sql import SparkSession
from pyspark.sql import Row
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

def getPipeline(goal, excep, rel):
    executed = []
    dependencies = [goal]
    while len(dependencies) != 0:
        # Remove all the dependencies of start node and executed node
        while (dependencies[0] in excep) or (dependencies[0] in executed):
            dependencies.pop(0)
            if len(dependencies) == 0:
                break
        if len(dependencies) == 0:
            continue
        dep = rel.lookup(dependencies[0])
        if len(dep) != 0:
            dependencies = dependencies + dep[0]
        executed.append(dependencies.pop(0))
    return executed

def main():
    spark = SparkSession.builder.master('spark://master:7077').appName("EQWork").getOrCreate()
    POIdf = spark.read.format("csv").option("header", "true").load("/tmp/data/POIList.csv")
    sampleDf = spark.read.format("csv").option("header", "true").load("/tmp/data/DataSample.csv")
    # 1) Filtered the identical geopoin and time
    cleanSample = sampleDf.dropDuplicates([' TimeSt','Latitude','Longitude'])
    poiDf = POIdf.drop_duplicates([' Latitude','Longitude'])
    poiList = poiDf.rdd.collect()
    # 2) Assign label to geopoint by the nearest POI
    labelSample = cleanSample.rdd.map(lambda r: calDist(r,poiList)).toDF()
    # 3) 1.Calculate the average and standar deviation
    # Calculate the average
    labelSample.groupBy('label').agg({'distance':'avg'})\
        .coalesce(1).write.format("csv").option("header", "true")\
        .mode("append").save("/tmp/data/Average")
    # Calculate the standard deviation
    labelSample.groupBy('label').agg({'distance':'stddev'})\
        .coalesce(1).write.format("csv").option("header", "true")\
        .mode("append").save("/tmp/data/StandardDeviation")
    # 3) 2.Calculate the radius and density of each POI
    radius = labelSample.groupBy('label').agg({'distance':'max'}).orderBy('label').collect()
    total = labelSample.groupBy('label').count().orderBy('label').collect()
    dense = [(total[x]['label'],total[x]['count']/(math.pi * radius[x]['max(distance)']**2)) for x in range(3)]
    # 4) Calculate the pipeline
    # Create relations list which reverse the order (dst,src)
    relations = spark.read.text('/tmp/data/relations.txt').rdd\
    .map(lambda l : (l[0].rstrip().split('->')[1],[l[0].split('->')[0]]))\
    .reduceByKey(lambda x,y : x + y)

    with open('/tmp/data/question.txt') as f:
        content = f.readlines()
    start = content[0].rstrip().split(' ')[2]
    goal = content[1].rstrip().split(' ')[2]
    # Generate the pipeline for start and goal point
    depenStart = getPipeline(start,[''],relations)
    depenGoal = getPipeline(goal,depenStart,relations)
    # reverse the order of list to get the right executed order
    depenGoal.reverse()
    # Write output to file
    cleanSample.coalesce(1).write.format("csv").option("header", "true").mode("append").save("/tmp/data/FilteredData")
    labelSample.coalesce(1).write.format("csv").option("header", "true").mode("append").save("/tmp/data/LabeledData")
    with open('/tmp/data/radius.txt','w') as f:
        for poi in radius:
            f.write(poi[0] + ": " + str(poi[1])+"\n")
    with open('/tmp/data/dense.txt','w') as f:
        for poi in dense:
            f.write(poi[0] + ": " + str(poi[1])+"\n")
    with open('/tmp/data/pipeline.txt','w') as f:
        f.write(str(depenGoal))

if __name__ == '__main__':
    main()
