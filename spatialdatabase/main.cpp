#include <string>
#include <set>
#include <map>
#include <iostream>
#include "database.h"

int main()
{
    DataBase db;
    TableDescription table;
    table.setName("test1");
    table.setSpatialKey("Область", SpatialType::POLYGON);
    table.setTemporalKey("Время", TemporalType::BITEMPORAL_TIME);
    AttributeDescription d1;
    d1.name = "col";
    d1.type = Type::REAL;
    table.addColumn(d1);
    table.setName("test2");
    std::cout <<db.createTable(table) << std::endl;
    std::cout <<db.createTable(table) << std::endl;

    return 0;
}
