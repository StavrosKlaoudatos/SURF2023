import sys, os
import duckdb
import pyarrow.parquet as pq
import duckdb

jobs = []
files= []
path = '/Users/stavrosklaoudatos/Desktop/HiggsData//Users/stavrosklaoudatos/Desktop/HiggsData/output_test2/'

for i in range(len(os.dirlist(path))): jobs.append('/job_{}'.format(str(i)))


for job in jobs:
   
    dir = os.listdir(path+job)

    for file in dir:
        if '.parquet' in file:
            files.append(path+job+'/'+file)


with pq.ParquetWriter("output.parquet", schema=pq.ParquetFile(files[0]).schema_arrow) as writer:
    for file in files:
        writer.write_table(pq.read_table(file))
