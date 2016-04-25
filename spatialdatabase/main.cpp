#include <string>
#include <set>
#include <map>
#include <iostream>
#include "database.h"
#include "hlbvh.h"

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

gpudb::AABB globalAABB;
bool init;

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

    aabb.x.x = std::min(x1, x2);
    aabb.y.x = std::min(y1, y2);
    aabb.z.x = std::min(z1, z2);
    aabb.w.x = std::min(w1, w2);

    aabb.x.y = std::max(x1, x2);
    aabb.y.y = std::max(y1, y2);
    aabb.z.y = std::max(z1, z2);
    aabb.w.y = std::max(w1, w2);

    if (!init) {
        init = true;
        globalAABB = aabb;
    }

    globalAABB.x.x = std::min(globalAABB.x.x, aabb.x.x);
    globalAABB.y.x = std::min(globalAABB.y.x, aabb.y.x);
    globalAABB.z.x = std::min(globalAABB.z.x, aabb.z.x);
    globalAABB.w.x = std::min(globalAABB.w.x, aabb.w.x);

    globalAABB.x.y = std::max(globalAABB.x.y, aabb.x.y);
    globalAABB.y.y = std::max(globalAABB.y.y, aabb.y.y);
    globalAABB.z.y = std::max(globalAABB.z.y, aabb.z.y);
    globalAABB.w.y = std::max(globalAABB.w.y, aabb.w.y);

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
    printf("Global AABB {%f %f} {%f %f} {%f %f} {%f %f}\n",
           globalAABB.x.x, globalAABB.x.y, globalAABB.y.x, globalAABB.y.y,
           globalAABB.z.x, globalAABB.z.y, globalAABB.w.x, globalAABB.w.y);
    cudaMemcpy(aabb, cpuAAABB, sizeof(gpudb::AABB) * size, cudaMemcpyHostToDevice);
    bvh.build(aabb, size);

    StackAllocator::getInstance().popPosition();
    gpudb::GpuStackAllocator::getInstance().popPosition();
    db.deinit();
    return 0;
}
