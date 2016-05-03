#include <string>
#include <set>
#include <map>
#include <iostream>
#include "timer.h"
#include "database.h"
#include "hlbvh.h"

gpudb::AABB globalAABB;
bool init;


int main()
{
    //srand(time(0));
    DataBase &db = DataBase::getInstance();
    Log::getInstance().showFilePathLevel(2);

    TableDescription table;
    table.setName("test1");
    table.setSpatialKey("Точка", SpatialType::POINT);
    table.setTemporalKey("Время", TemporalType::VALID_TIME);
    AttributeDescription desc;
    desc.name = "col";
    desc.type = Type::REAL;
    table.addColumn(desc);
    desc.name = "col2";
    table.addColumn(desc);

    Date date;
    date.set(2016, 12, 31, 5, 23, 48, 453789);
    Date date2;
    date2.setFromCode(date.codeDate());
    date2.setDate(2017, 12, 12);
    Attribute atr;
    atr.setName("col");
    atr.setValue(false, 0.0f);
    Attribute atr2;
    atr2.setName("col2");
    atr2.setValue(false, 0.0f);
    Row newRow;
    newRow.spatialKey.name = "Точка";
    newRow.temporalKey.name = "Время";
    newRow.spatialKey.type = SpatialType::POINT;
    newRow.temporalKey.type = TemporalType::VALID_TIME;
    newRow.temporalKey.validTimeS = date;
    newRow.temporalKey.validTimeE = date2;
    newRow.spatialKey.points.push_back(make_float2(0.5, 0));
    newRow.addAttribute(atr);
    newRow.addAttribute(atr2);

    db.createTable(table);
    db.insertRow("test1", newRow);
    newRow.spatialKey.points.clear();
    newRow.spatialKey.points.push_back(make_float2(0.5, 0.0));
    db.insertRow("test1", newRow);
    newRow.spatialKey.points.clear();
    newRow.spatialKey.points.push_back(make_float2(0.755, 0.0));
    db.insertRow("test1", newRow);
    newRow.spatialKey.points.clear();
    newRow.spatialKey.points.push_back(make_float2(0.42, 0.0));
    db.insertRow("test1", newRow);
    newRow.spatialKey.points.clear();
    newRow.spatialKey.points.push_back(make_float2(0.1241, 0.0));
    db.insertRow("test1", newRow);
    newRow.spatialKey.points.clear();
    newRow.spatialKey.points.push_back(make_float2(0.54, 0.0));
    db.insertRow("test1", newRow);

    std::vector<float2> random;
    for (int i = 0; i < 20000; i++) {
        newRow.spatialKey.points.clear();
        newRow.spatialKey.points.push_back(make_float2(rand() / float(RAND_MAX) * 10, rand() / float(RAND_MAX)));
        random.push_back(newRow.spatialKey.points[0]);
        db.insertRow("test1", newRow);
    }

    table.setName("test2");
    db.createTable(table);
    newRow.spatialKey.points.clear();
    newRow.spatialKey.points.push_back(make_float2(4.031103, 0.169586));
    db.insertRow("test2", newRow);
    TempTable temptable1;
    std::cout << db.selectTable("test1", temptable1) << std::endl;


    TempTable temptable2, temptable4;
    std::cout << db.selectTable("test2", temptable2) << std::endl;
    pointxpointKnearestNeighbor(temptable1, temptable1, 24, temptable4);
    temptable4.showTable();
    //temptable1.showTable();
    //temptable1.showTable();
   // temptable2.showTable();

    //pointxpointKnearestNeighbor(temptable2, temptable1, 24, temptable3);
  //  pointxpointKnearestNeighbor(temptable2, temptable1, 24, temptable4);

   // temptable3.showTable();
    //temptable2.showTable();

    //pointxpointKnearestNeighbor(TempTable &a, TempTable &b, TempTable &result);

    /*gpudb::HLBVH bvh;
    gpudb::GpuStackAllocator::getInstance().pushPosition();
    const int size =  8000000;
    gpudb::AABB *aabb = gpudb::GpuStackAllocator::getInstance().alloc<gpudb::AABB>(size);
    StackAllocator::getInstance().pushPosition();
    gpudb::AABB *cpuAAABB = StackAllocator::getInstance().alloc<gpudb::AABB>(size);

    for (int i = 0; i < size; i++) {
        cpuAAABB[i] = genAABB();
    }
    printf("Global AABB {%f %f} {%f %f} {%f %f} {%f %f}\n",
           globalAABB.x.x, globalAABB.x.y, globalAABB.y.x, globalAABB.y.y,
           globalAABB.z.x, globalAABB.z.y, globalAABB.w.x, globalAABB.w.y);
    cudaMemcpy(aabb, cpuAAABB, sizeof(gpudb::AABB) * size, cudaMemcpyHostToDevice);
    Timer t;
    t.start();
    bvh.build(aabb, size);
    cudaDeviceSynchronize();
    printf("bvh build %d ms\n", t.elapsedMillisecondsU64());

    t.start();
    for (int i = 1000; i < 1001; i++) {
        if (!bvh.search(cpuAAABB[i])) {
            printf("%d\n ", 0);
        }
    }
    cudaDeviceSynchronize();
    printf("bvh search %d ms %d levels\n", t.elapsedMillisecondsU64(), bvh.numBVHLevels);

    bvh.free();

    StackAllocator::getInstance().popPosition();
    gpudb::GpuStackAllocator::getInstance().popPosition();*/
    db.deinit();

    return 0;
}
