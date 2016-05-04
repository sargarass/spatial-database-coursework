#include "testrequests.h"

void test() {
    DataBase &db = DataBase::getInstance();

    TableDescription table;
    table.setName("test1");
    table.setSpatialKey("Точка", SpatialType::POINT);
    table.setTemporalKey("Время", TemporalType::BITEMPORAL_TIME);
    AttributeDescription desc;
    desc.name = "col";
    desc.type = Type::REAL;
    table.addColumn(desc);
    desc.type = Type::STRING;
    desc.name = "col2";
    table.addColumn(desc);

    Date date;
    date.set(2016, 12, 31, 5, 23, 48, 453789);
    Date date2;
    date2.setFromCode(date.codeDate());
    date2.setDate(2017, 12, 12);
    Attribute atr;
    atr.setName("col");
    atr.setValue(false, 5.0f);
    Attribute atr2;
    atr2.setName("col2");
    atr2.setValue(false, "PLEASE EXCUSE MY FRENCH");
    Row newRow;
    newRow.spatialKey.name = "Точка";
    newRow.temporalKey.name = "Время";
    newRow.spatialKey.type = SpatialType::POINT;
    newRow.temporalKey.type = TemporalType::BITEMPORAL_TIME;
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
    for (int i = 0; i < 30; i++) {
        Date d1, d2;
        d1 = Date::getRandomDate();
        d2 = Date::getRandomDate();
        if (d1.codeDate() > d2.codeDate()) {
            std::swap(d1, d2);
        }

        newRow.temporalKey.validTimeS = d1;
        newRow.temporalKey.validTimeE = d2;
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

    std::set<Attribute> atrSet;
    atr2.setValue(false, "MOSCOW NEVER CRY");
    atr.setValue(false, 0.0);
    atrSet.insert(atr2);
    atrSet.insert(atr);


    Predicate p = getTesterPointer();
    db.showTable("test1");

    db.update("test1", atrSet, p);

    TempTable temptable1;
    std::cout << db.selectTable("test1", temptable1) << std::endl;
    db.showTable("test1");
    db.dropRow("test1", p);
    db.showTable("test1");
    db.insertRow("test1", newRow);
    db.showTable("test1");

    TempTable temptable2, temptable4, temptable5, temptable6;
    db.selectTable("test2", temptable2);
    pointxpointKnearestNeighbor(temptable1, temptable1, 24, temptable4);
    //pointxpointKnearestNeighbor(temptable4, temptable4, 24, temptable5);
   // temptable5.showTable();
    //pointxpointKnearestNeighbor(temptable5, temptable5, 24, temptable6);
    //temptable6.showTable();
}
