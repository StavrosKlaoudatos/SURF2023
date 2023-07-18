import sys, os
import duckdb
import pyarrow.parquet as pq
import duckdb



jobs = []
files= []
names =[]
path = '/Users/stavrosklaoudatos/Desktop/HiggsData/output_test2/'




dirs = os.listdir(path)

print(dirs)


for dir in dirs:

    files= []

    if 'DS_Store' in dir:
        pass
    else:
        epath = path + dir
        names.append(dir)

        for job in os.listdir(epath):
            jdir = os.listdir(epath +'/'+job)
            

            for file in jdir:
                if '.parquet' in file:
                    files.append(epath+ '/'+job+'/'+file)

        
        print('======================== \n', dir)
        with pq.ParquetWriter("{}.parquet".format(dir), schema=pq.ParquetFile(files[0]).schema_arrow) as writer:
            for file in files:
                writer.write_table(pq.read_table(file))
        

        

