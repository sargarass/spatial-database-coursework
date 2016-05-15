#include "testrequests.h"

void test() {
    DataBase &db = DataBase::getInstance();
    std::cout << db.loadFromDisk("/home/sargarass/tmp/db/test.sdata") << std::endl;

    TableDescription table;
    table.setName("test1");
    table.setSpatialKey("Точка", SpatialType::POINT);
    table.setTemporalKey("Время", TemporalType::VALID_TIME);
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
    atr.setNullValue(Type::REAL);
    Attribute atr2;
    atr2.setName("col2");
    atr2.setValue("PLEASE EXCUSE MY FRENCH");

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
    std::vector<Row> inserted;
    //inserted.push_back(newRow);

    //db.createTable(table);
   // db.insertRow("test1", newRow);
   // newRow.spatialKey.points.clear();
   // newRow.spatialKey.points.push_back(make_float2(0.5, 1.1));
   // inserted.push_back(newRow);
  //  db.insertRow("test1", newRow);
   // newRow.spatialKey.points.clear();
  //  newRow.spatialKey.points.push_back(make_float2(0.755, 0.0));
  //  inserted.push_back(newRow);
  //  db.insertRow("test1", newRow);
  //  newRow.spatialKey.points.clear();
  //  newRow.spatialKey.points.push_back(make_float2(0.42, 0.0));
  //  inserted.push_back(newRow);
   // db.insertRow("test1", newRow);
  //  newRow.spatialKey.points.clear();
  //  newRow.spatialKey.points.push_back(make_float2(0.1241, 0.0));
 //   inserted.push_back(newRow);
  //  db.insertRow("test1", newRow);
 //   newRow.spatialKey.points.clear();
 //   newRow.spatialKey.points.push_back(make_float2(0.54, 0.0));
 //   inserted.push_back(newRow);
   // db.insertRow("test1", newRow);

/*
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
        newRow.spatialKey.points.push_back(make_float2(rand() / float(RAND_MAX) * 10, rand() / float(RAND_MAX) * 10));
        random.push_back(newRow.spatialKey.points[0]);
        //db.insertRow("test1", newRow);
        inserted.push_back(newRow);
    }
*/

    /*table.setName("test2");
    table.setSpatialKey("Полигонъ", SpatialType::POLYGON);
    db.createTable(table);

    newRow.spatialKey.points.clear();
    newRow.spatialKey.points.push_back(make_float2(0.0, -1.0));
    newRow.spatialKey.points.push_back(make_float2(1.0, 0.0));
    newRow.spatialKey.points.push_back(make_float2(0.0, 10));
    newRow.spatialKey.points.push_back(make_float2(-1.0, 0.0));
    newRow.spatialKey.name = "Полигонъ";
    newRow.spatialKey.type = SpatialType::POLYGON;
    db.insertRow("test2", newRow);

    //db.insertRow("test2", newRow);
    db.showTable("test1");
    std::set<Attribute> atrSet;
    atr2.setValue("MOSCOW NEVER CRY");
    atr.setValue(4542.5489);
    atrSet.insert(atr2);
    atrSet.insert(atr);
*/

    Predicate p = getTesterPointer();
    //db.showTable("test1");
/*
    db.dropTable("test3");

    TableDescription lineTable;
    lineTable.setName("test3");
    lineTable.setSpatialKey("Линия", SpatialType::LINE);
    lineTable.setTemporalKey("Время", TemporalType::VALID_TIME);
    AttributeDescription lineTableAtrDesc;
    lineTableAtrDesc.name = "col";
    lineTableAtrDesc.type = Type::REAL;
    lineTable.addColumn(lineTableAtrDesc);
    lineTableAtrDesc.type = Type::STRING;
    lineTableAtrDesc.name = "col2";
    lineTable.addColumn(lineTableAtrDesc);
    db.createTable(lineTable);
    //db.showTableHeader("test3");

    newRow.spatialKey.points.clear();
    newRow.spatialKey.points.push_back(make_float2(0.0, 0.0));
    newRow.spatialKey.points.push_back(make_float2(7.0, 0.0));
    newRow.spatialKey.points.push_back(make_float2(7.0, 6.0));
    newRow.spatialKey.name = "Линия";
    newRow.spatialKey.type = SpatialType::LINE;
    db.insertRow("test3", newRow);
    newRow.spatialKey.points.clear();
    newRow.spatialKey.points.push_back(make_float2(0.0, -20.0));
    newRow.spatialKey.points.push_back(make_float2(0.0, -25.0));
    newRow.spatialKey.points.push_back(make_float2(-7.0, -25.0));
    newRow.spatialKey.name = "Линия";
    newRow.spatialKey.type = SpatialType::LINE;
    db.insertRow("test3", newRow);

    newRow.spatialKey.points.clear();
    newRow.spatialKey.points.push_back(make_float2(0.0, 1.0));
    newRow.spatialKey.points.push_back(make_float2(0.0, 5.0));
    newRow.spatialKey.points.push_back(make_float2(-7.0, 5.0));
    newRow.spatialKey.name = "Линия";
    newRow.spatialKey.type = SpatialType::LINE;
    db.insertRow("test3", newRow);
*/
    TempTable temptable1, temptable2, temptable3;
    db.selectTable("test1", temptable1);
    db.selectTable("test2", temptable2);
    db.selectTable("test3", temptable3);

   /* std::vector<Row> selected;
    db.selectRow(temptable1, p, selected);

    for (int i = 0; i < selected.size(); i++) {
        std::cout << selected[i].temporalKey.validTimeS.toString() << " " << selected[i].temporalKey.validTimeE.toString() << std::endl;
        std::cout << selected[i].getAttributeSize() << std::endl;

        for (int j = 0; j < selected[i].getAttributeSize(); j++) {
            std::cout << selected[i].getAttribute(j).getName() << " : "
            << typeToString(selected[i].getAttribute(j).getType()) << " : ";
            if (!selected[i].getAttribute(j).isNull()) {
                switch (selected[i].getAttribute(j).getType()) {
                    case Type::REAL:
                        std::cout << selected[i].getAttribute(j).getReal() << " ";
                        break;
                    case Type::STRING:
                        std::cout << selected[i].getAttribute(j).getString() << " ";
                        break;
                    case Type::INT:
                        std::cout << selected[i].getAttribute(j).getInt() << " ";
                        break;
                }
            } else {
                std::cout << "NULL" << std::endl;
            }
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }*/
    //db.showTable(temptable1);
//    db.polygonxpointPointsInPolygon(temptable2, temptable1);
//    //db.showTable("test1");
//   // db.showTable("test2");
//    std::cout << "________________________" <<std::endl;

//    //db.dropRow("test1", p);
//    //db.showTable("test1");
//  //  db.insertRow("test1", newRow);
//   // db.showTable("test1");

// //   TempTable temptable5;
    //db.selectTable("test2", temptable2);
   // db.selectTable("test3", temptable3);
    db.showTable("test2");

    TempTable &&temptable5 = db.pointxpointKnearestNeighbor(temptable1, temptable1, 24);
    db.showTable(temptable5);

    TempTable &&temptable4 = db.linexpointPointsInBufferLine(temptable3, temptable1, 1.0f);
    db.showTable(temptable4);

    TempTable &&temptable6 = db.polygonxpointPointsInPolygon(temptable2, temptable1);
    db.showTable(temptable6);

    TempTable &&temptable7 = db.polygonxpointPointsInPolygon(temptable6, temptable1);
    db.showTable(temptable7);

//    std::vector<Row> selected;
//    db.selectRow(temptable4, p, selected);

//    for (int i = 0; i < selected.size(); i++) {
//        std::cout << selected[i].temporalKey.validTimeS.toString() << " " << selected[i].temporalKey.validTimeE.toString() << std::endl;
//        std::cout << selected[i].getAttributeSize() << std::endl;

//        for (int j = 0; j < selected[i].getAttributeSize(); j++) {
//            std::cout << selected[i].getAttribute(j).getName() << " : "
//            << typeToString(selected[i].getAttribute(j).getType()) << " : ";
//            if (!selected[i].getAttribute(j).isNull()) {
//                switch (selected[i].getAttribute(j).getType()) {
//                    case Type::REAL:
//                        std::cout << selected[i].getAttribute(j).getReal() << " ";
//                        break;
//                    case Type::STRING:
//                        std::cout << selected[i].getAttribute(j).getString() << " ";
//                        break;
//                    case Type::INT:
//                        std::cout << selected[i].getAttribute(j).getInt() << " ";
//                        break;
//                    case Type::SET:
//                        db.showTable(*selected[i].getAttribute(j).getSet());
//                        break;
//                }
//            } else {
//                std::cout << "NULL";
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//        std::cout << std::endl;
//    }
        std::cout << db.insertRow("test1", inserted) << std::endl;
    //db.update(temptable4, atrSet, p);
  // std::cout << db.showTable(temptable4) << std::endl;
  //  std::cout << "________________________" <<std::endl;
  //  db.update(temptable4, atrSet, p);

  //  std::cout << "________________________" <<std::endl;
  //  db.showTable(temptable4);
   // std::cout << "________________________" <<std::endl;
   // db.dropRow(temptable4, p);    //db.showTable(temptable4);
   // std::cout << "________________________" <<std::endl;
   // db.showTable(temptable4);

    //pointxpointKnearestNeighbor(temptable4, temptable4, 24, temptable5);
   // temptable5.showTable();
    //pointxpointKnearestNeighbor(temptable5, temptable5, 24, temptable6);
    //temptable6.showTable();
    std::cout << db.saveOnDisk("/home/sargarass/tmp/db/test.sdata") << std::endl;
    //std::cout << db.loadFromDisk("/home/sargarass/tmp/db/test.sdata") << std::endl;
}
