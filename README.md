# spatial-database-coursework
The database is allowed to create tables with spatial-temporal keys and values of different types. Every key is unique in a table.
The key consists of two parts:
1. Spatial part is a polygon or point or polyline in 2D (each point are in \[-180;180\]X\[-90;90\]).
2. Temporal part is interval of *valid time* or *transaction time* or *bitemporal time* (both *valid time* and *transaction time*)
Values is a string where each element can be one of the following types: *string*, *integer*, *float*, *time*.
The Database is not supported parallel queries. 
Each query to the database is simultaneously applied to all rows (1 Cuda thread per row) and is accelerated by [HLBVH2](https://dl.acm.org/citation.cfm?id=2018333). It is a spatial binary tree which can be built on GPU in linear time. It used z-curve for mapping 4D space (2D spatial + 2D temporal spaces) to the 1D 96bit key. 

The database supports the following types of operation:

1. Find k-nearest neighbours for points.

> **Input**: two tables with points as spatial part of keys
>
> **Output**: new table with rows of the first table which are modified by addition column. Each element in this column consists of the rows from the second table that are k-nearest neighbours for the key in the row.
>
>Algorithm used HLBVH2 and based on computing [min/minimax dist](http://postgis.refractions.net/support/nearestneighbor.pdf)

2. Find points that are laid in polygon:

> **Input**: two tables, the first with polygon as spatial part of keys and the second with point.
>
> **Output**: new table with rows of the first table which are modified by addition column. Each element in this column consists of the rows from the second table such that the point from the key in the row from the second table are laid in polygon from the key in the row from the first table.

3. Find points that are located on distance not more than *R* from the polyline.

> **Input**: two tables, the first with polyline as spatial part of keys and the second with point.
>
> **Output**: new table with rows of the first table which are modified by addition column. Each element in this column consists of the rows from the second table such that the point from the key in the row from the second table are located on distance not more than *R* from the polyline from the key in the row from the first table.

4. Insert many rows in a table as one query.
5. Delete, update, select rows from table by user-specified predicate.

For more information please read [texv2.pdf](https://github.com/sargarass/spatial-database-coursework/blob/master/texv2.pdf) (in russian)

# Libraries that were used:
1. cmake 3.0
2. CUDA (nvidia-cuda-toolkit)
3. glew
4. glfw3
5. thrust
6. cub
7. moderngpu
8. openssl

# How to build (tested in linux gentoo)
In the folder with project type in console:
- cmake CMakeLists.txt
- make

# How to run
You need nvidia videocard which are support compute capability 2.0.

In the *bin* folder run console application *DataBase*

# Cache tuning
In the file *database.cpp* you can change cache size in gpu and ram. 

These caches are used to fast allocation temp memories for operation (stack allocators).
```C++
gpudb::GpuStackAllocator::getInstance().resize(1056ULL * 1024ULL * 1024ULL);
StackAllocator::getInstance().resize(1024ULL * 1024ULL * 1024ULL);
```
