#include <string>
#include <set>
#include <map>
#include <iostream>
#include "database.h"
#include "hlbvh.h"

float getMax(float a, float b) {
    if (a > b) {
        return a;
    }
    return b;
}

float getMin(float a, float b){
    if (a > b) {
        return b;
    }
    return a;
}

float clamp(float in, float min, float max) {
    float res = in;
    if (res > max) {
        res = max;
    }
    if (res < min) {
        res = min;
    }
    return res;
}
gpudb::AABB genAABB() {
    gpudb::AABB aabb;
    float x1 = clamp((rand() / (float)RAND_MAX) * 360.0f - 180.0f, -180.0f, 180.0f);
    float x2 = clamp((rand() / (float)RAND_MAX) * 360.0f - 180.0f, -180.0f, 180.0f);
    float y1 = clamp((rand() / (float)RAND_MAX) * 180.0f - 90.0f, -90.0f, 90.0f);
    float y2 = clamp((rand() / (float)RAND_MAX) * 180.0f - 90.0f, -90.0f, 90.0f);
    float z1 = clamp((rand() / (float)RAND_MAX), 0.0f, 1.0f);
    float z2 = clamp((rand() / (float)RAND_MAX), 0.0f, 1.0f);
    float w1 = clamp((rand() / (float)RAND_MAX), 0.0f, 1.0f);
    float w2 = clamp((rand() / (float)RAND_MAX), 0.0f, 1.0f);

    aabb.x.x = getMin(x1, x2);
    aabb.y.x = getMin(y1, y2);
    aabb.z.x = getMin(z1, z2);
    aabb.w.x = getMin(w1, w2);

    aabb.x.y = getMax(x1, x2);
    aabb.y.y = getMax(y1,  y2);
    aabb.z.y = getMax(z1, z2);
    aabb.w.y = getMax(w1, w2);
    aabb.numComp = 4;
    return aabb;
}

int main()
{
    DataBase &db = DataBase::getInstance();
    TableDescription table;
    table.setName("test1");
    table.setSpatialKey("Область", SpatialType::POLYGON);
    table.setTemporalKey("Время", TemporalType::BITEMPORAL_TIME);
    AttributeDescription d1;
    d1.name = "col";
    d1.type = Type::STRING;
    table.addColumn(d1);
    AttributeDescription d2;
    d2.name = "col2";
    d2.type = Type::INT;
    table.addColumn(d1);
    table.addColumn(d2);

    Log::getInstance().showFilePathLevel(2);
    Attribute v;
    v.setName("col");
    v.setValue(false, "this is string");
    Row newRow;
    Attribute v2;
    v2.setName("col2");
    v2.setValue(false, 77771);

    newRow.spatialKey.name = "Область";
    newRow.temporalKey.name = "Время";
    newRow.spatialKey.type = SpatialType::POLYGON;
    newRow.temporalKey.type = TemporalType::BITEMPORAL_TIME;
    newRow.values.insert(v);
    newRow.values.insert(v2);

    db.createTable(table);
    db.insertRow("test1", newRow);

    newRow.values.clear();
    v.setValue(true, "");
    v2.setValue(true, 0);
    newRow.values.insert(v);
    newRow.values.insert(v2);
    db.insertRow("test1", newRow);

    newRow.values.clear();
    v.setValue(true, "");
    v2.setValue(true, 0);
    newRow.values.insert(v);
    newRow.values.insert(v2);
    db.insertRow("test1", newRow);

    newRow.values.clear();
    v.setValue(true, "");
    v2.setValue(true, 0);
    newRow.values.insert(v);
    newRow.values.insert(v2);
    db.insertRow("test1", newRow);

    newRow.values.clear();
    v.setValue(false, "dddddd");
    v2.setValue(true, 0);
    newRow.values.insert(v);
    newRow.values.insert(v2);
    db.insertRow("test1", newRow);

    newRow.values.clear();
    v.setValue(true, "");
    v2.setValue(false, 0);
    newRow.values.insert(v);
    newRow.values.insert(v2);
    db.insertRow("test1", newRow);

    db.showTable("test1");
    gpudb::HLBVH bvh;
    gpudb::GpuStackAllocator::getInstance().pushPosition();
    const int size =  250000;
    gpudb::AABB *aabb = gpudb::GpuStackAllocator::getInstance().alloc<gpudb::AABB>(size);
    StackAllocator::getInstance().pushPosition();
    gpudb::AABB *cpuAAABB = StackAllocator::getInstance().alloc<gpudb::AABB>(size);

    for (int i = 0; i < size; i++) {
        cpuAAABB[i] = genAABB();
        /*printf("AABB {%f, %f} {%f, %f} {%f, %f} {%f, %f} \n", cpuAAABB[i].x.x, cpuAAABB[i].x.y,
               cpuAAABB[i].y.x, cpuAAABB[i].y.y,
               cpuAAABB[i].z.x, cpuAAABB[i].z.y,
               cpuAAABB[i].w.x, cpuAAABB[i].w.y);*/
    }

    cudaMemcpy(aabb, cpuAAABB, sizeof(gpudb::AABB) * size, cudaMemcpyHostToDevice);
    bvh.build(aabb, size);

    StackAllocator::getInstance().popPosition();
    gpudb::GpuStackAllocator::getInstance().popPosition();
    db.deinit();
    return 0;
}
