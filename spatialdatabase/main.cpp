#include "cuda_staff.h"
#include <string>
#include <set>
#include <map>
#include <iostream>
#include "database.h"

int main(int argc, char *argv[])
{
    DataBase db;
    TableDescription table;
    table.setName("test1");
    table.setSpatialKey(SpatialType::POLYGON);
    table.setTemporalKey(TemporalType::BITEMPORAL_TIME);
    AttributeDescription d1;
    d1.name = "col";
    d1.type = Type::REAL;
    table.setName("test2");
    std::cout <<db.createTable(table) << std::endl;
    std::cout <<db.createTable(table) << std::endl;

    return 0;
}
