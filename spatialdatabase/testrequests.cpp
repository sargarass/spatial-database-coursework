#include "testrequests.h"
#include <fstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

void reshape(int width, int height )
{
    glViewport( 0, 0, (GLint) width, (GLint) height );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho(-1.2,1.2,-1.2,1.2,-1,1);
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
}

void drawCircle(float x, float y, float r, int amountSegments)
{
    glBegin(GL_LINE_LOOP);

    for(int i = 0; i < amountSegments; i++)
    {
        float angle = 2.0 * 3.1415926 * float(i) / float(amountSegments);

        float dx = r * cosf(angle);
        float dy = r * sinf(angle);

        glVertex2f(x + dx, y + dy);
    }

    glEnd();
}


void pointline() {
    std::vector<std::string> names;
    std::vector<std::string> secondNames;
    std::ifstream names_fin("../имена.txt");
    std::ifstream secondNames_fin("../фамилии.txt");
    std::copy(std::istream_iterator<std::string>(names_fin),
              std::istream_iterator<std::string>(),
              std::back_inserter(names));

    std::copy(std::istream_iterator<std::string>(secondNames_fin),
              std::istream_iterator<std::string>(),
              std::back_inserter(secondNames));

    srand(621);
    DataBase &db = DataBase::getInstance();
    // create
    TableDescription roads, buildings;
    roads.setName("Дороги");
    roads.setSpatialKey("Дорога", SpatialType::LINE);
    roads.setTemporalKey("Время транзакции", TemporalType::TRANSACTION_TIME);

    buildings.setName("Строения");
    buildings.setSpatialKey("Позиция", SpatialType::POINT);
    buildings.setTemporalKey("Время транзакции", TemporalType::TRANSACTION_TIME);

    AttributeDescription roadName; roadName.type = Type::STRING;
    roadName.name = "Название"; roads.addColumn(roadName);
    AttributeDescription buildingName, buildingType;
    buildingName.type = Type::STRING; buildingName.name = "Название";
    buildingType.type = Type::STRING; buildingType.name = "Тип";
    buildings.addColumn(buildingName); buildings.addColumn(buildingType);

    db.createTable(roads);
    db.createTable(buildings);
    //
    db.showTableHeader("Дороги");
    db.showTableHeader("Строения");

    Attribute m4Atr;
    m4Atr.setName("Название");
    m4Atr.setValue("M4");
    Row m4; m4.addAttribute(m4Atr);
    m4.spatialKey.name = "Дорога";
    m4.spatialKey.type = SpatialType::LINE;
    m4.spatialKey.points = { make_float2(0.1, 0.1),
                                 make_float2(0.2, 0.1),
                                 make_float2(0.1, 0.2),
                                 make_float2(0.1, 0.3),
                                 make_float2(0.3, 0.4),
                                 make_float2(0.4, 0.0),
                                 make_float2(0.5, 0.0),
                                 make_float2(0.7, 0.1),
                                 make_float2(0.5, 0.4),
                                 make_float2(0.5, 0.6),
                                 make_float2(0.4, 0.6),
                                 make_float2(0.4, 0.7),
                                 make_float2(0.8, 0.7),
                                 make_float2(0.9, 0.0)};
    m4.temporalKey.name = "Время транзакции"; m4.temporalKey.type = TemporalType::TRANSACTION_TIME;

    db.insertRow("Дороги", m4);
    db.showTable("Дороги");

    std::vector<std::string> localTypes = {"Кинотеатр", "Кафе", "Ресторан", "Отель", "Шаурма", "Церковь \"Ктулху\"", "Стадион", "Памятник"};

    for (int i = 0; i < 35; i++) {
        Row newBuilding;
        newBuilding.spatialKey.name = "Позиция"; newBuilding.spatialKey.type = SpatialType::POINT;
        newBuilding.spatialKey.points.push_back(make_float2(((rand() / ((double)RAND_MAX))), ((rand() / ((double)RAND_MAX)))));
        newBuilding.temporalKey.name = "Время транзакции"; newBuilding.temporalKey.type = TemporalType::TRANSACTION_TIME;
        Attribute name, type;
        name.setName("Название");
        name.setValue("Улица \"" + names[((rand() / ((double)RAND_MAX))) * names.size()] + "\"");
        type.setName("Тип");
        type.setValue(localTypes[((rand() / ((double)RAND_MAX))) * localTypes.size()]);
        newBuilding.addAttribute(name);
        newBuilding.addAttribute(type);
        db.insertRow("Строения", newBuilding);
    }
    db.showTable("Строения");

    std::vector<Row> roadRows = db.selectRow("Дороги", SELECT_ALL_ROWS()).unwrap();
    auto buildingTT = db.selectTable("Строения").unwrap();
    auto roadTT = db.selectTable("Дороги").unwrap();
    double radious = 0.05;
    auto outTT = db.linexpointPointsInBufferLine(roadTT, buildingTT, radious).unwrap();
    std::vector<Row> buildingRows = db.selectRow("Строения", SELECT_ALL_ROWS()).unwrap();
    std::vector<Row> linexpointRows = db.selectRow(outTT, SELECT_ALL_ROWS()).unwrap();


    db.showTable(outTT);
    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        exit( EXIT_FAILURE );
    }
    glewExperimental = GL_TRUE;
    glewInit();
    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* window = glfwCreateWindow (1000, 1000, "Hello Triangle", NULL, NULL);
    if (!window) {
        fprintf (stderr, "ERROR: could not open window with GLFW3\n");
        glfwTerminate();
    }
    glfwMakeContextCurrent (window);
    reshape(1000, 1000);

    auto set = linexpointRows[0].getAttribute(1).unwrap().getSet().unwrap();
    auto setfiltered = db.filter(set, roadFilter()).unwrap();
    db.showTable(setfiltered);
    auto setRows = db.selectRow(set, SELECT_ALL_ROWS()).unwrap();
    auto setRowsFiltered = db.selectRow(setfiltered, SELECT_ALL_ROWS()).unwrap();

    do{
        glClearColor(1, 1, 1, 1);
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        reshape(width, height);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        glTranslatef(-1.0f, -1.0f, 0.0f);
        glScalef(2, 2, 1);
        glPointSize(10.0);
        glLineWidth(2.0);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH_HINT,  GL_NICEST);
        for (uint i = 0; i < roadRows.size(); i++) {
            glBegin(GL_LINE_STRIP);
            for (uint j = 0; j < roadRows[i].spatialKey.points.size(); j++) {
                glColor3f(0, 0, 1);
                glVertex2f(roadRows[i].spatialKey.points[j].x, roadRows[i].spatialKey.points[j].y);
            }
            glEnd();
        }
        // отрисовка lineBuffer
        for (uint i = 0; i < roadRows.size(); i++) {
            for (uint j = 0; j < roadRows[i].spatialKey.points.size(); j++) {
                glColor3f(0, 1, 0);
                drawCircle(roadRows[i].spatialKey.points[j].x, roadRows[i].spatialKey.points[j].y, radious, 32);

                if (j + 1 < roadRows[i].spatialKey.points.size()) {
                    float2 a = roadRows[i].spatialKey.points[j];
                    float2 b = roadRows[i].spatialKey.points[j + 1];
                    float2 n = b - a; n = norma(n);
                    n = make_float2(n.y, -n.x);
                    float2 v1 = a + n * radious;
                    float2 v2 = b + n * radious;
                    float2 v3 = b - n * radious;
                    float2 v4 = a - n * radious;
                    glBegin(GL_LINES);
                        glVertex2f(v1.x, v1.y);
                        glVertex2f(v2.x, v2.y);
                        glVertex2f(v3.x, v3.y);
                        glVertex2f(v4.x, v4.y);
                    glEnd();
                }
            }
        }

        for (uint i = 0; i < buildingRows.size(); i++) {
            glBegin(GL_POINTS);
            for (uint j = 0; j < buildingRows[i].spatialKey.points.size(); j++) {
                glColor3f(0, 0, 0);
                glVertex2f(buildingRows[i].spatialKey.points[j].x, buildingRows[i].spatialKey.points[j].y);
            }
            glEnd();
        }

        for (uint i = 0; i < buildingRows.size(); i++) {
            glBegin(GL_POINTS);
            for (uint j = 0; j < buildingRows[i].spatialKey.points.size(); j++) {
                glColor3f(0, 0, 0);
                glVertex2f(buildingRows[i].spatialKey.points[j].x, buildingRows[i].spatialKey.points[j].y);
            }
            glEnd();
        }



        for (uint i = 0; i < setRows.size(); i++) {
            glBegin(GL_POINTS);
                glColor3f(1, 0, 0);
                glVertex2f(setRows[i].spatialKey.points[0].x, setRows[i].spatialKey.points[0].y);
            glEnd();
        }

        for (uint i = 0; i < setRowsFiltered.size(); i++) {
            glBegin(GL_POINTS);
                glColor3f(0.8, 0.0, 0.8);
                glVertex2f(setRowsFiltered[i].spatialKey.points[0].x, setRowsFiltered[i].spatialKey.points[0].y);
            glEnd();
        }



        glfwSwapBuffers(window);
        glfwPollEvents();

    }
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose(window) == 0 );
}

void lygynika() {
    srand(0);
    DataBase &db = DataBase::getInstance();

    TableDescription abonents, offices;
    abonents.setName("Здания Лугуники");
    abonents.setSpatialKey("Позиция", SpatialType::POINT);
    abonents.setTemporalKey("Время", TemporalType::VALID_TIME);
    AttributeDescription name, secondName;
    name.name = "Имя"; name.type = Type::STRING;
    secondName.name = "Фамилия"; secondName.type = Type::STRING;
    abonents.addColumn(name); abonents.addColumn(secondName);

    offices.setName("Офисы ТЕЛЕ-3");
    offices.setSpatialKey("Граница офиса", SpatialType::POLYGON);
    offices.setTemporalKey("Время внесения в таблицу", TemporalType::TRANSACTION_TIME);
    AttributeDescription street, officeName;
    street.name = "Адрес"; street.type = Type::STRING;
    officeName.name = "Название офиса"; officeName.type = Type::STRING;
    offices.addColumn(street); offices.addColumn(officeName);
    db.createTable(abonents);
    db.createTable(offices);
    db.showTableHeader("Абоненты ТЕЛЕ-3");
    db.showTableHeader("Офисы ТЕЛЕ-3");
}

void tele3() {
    std::vector<std::string> names;
    std::vector<std::string> secondNames;
    std::ifstream names_fin("../имена.txt");
    std::ifstream secondNames_fin("../фамилии.txt");
    std::copy(std::istream_iterator<std::string>(names_fin),
              std::istream_iterator<std::string>(),
              std::back_inserter(names));

    std::copy(std::istream_iterator<std::string>(secondNames_fin),
              std::istream_iterator<std::string>(),
              std::back_inserter(secondNames));

    srand(0);
    DataBase &db = DataBase::getInstance();
    std::vector<Row> result;
    {
        {
            TableDescription abonents, offices;
            abonents.setName("Абоненты ТЕЛЕ-3");
            abonents.setSpatialKey("Позиция", SpatialType::POINT);
            abonents.setTemporalKey("Время", TemporalType::VALID_TIME);
            AttributeDescription name, secondName;
            name.name = "Имя"; name.type = Type::STRING;
            secondName.name = "Фамилия"; secondName.type = Type::STRING;
            abonents.addColumn(name); abonents.addColumn(secondName);

            offices.setName("Офисы ТЕЛЕ-3");
            offices.setSpatialKey("Граница офиса", SpatialType::POLYGON);
            offices.setTemporalKey("Время внесения в таблицу", TemporalType::TRANSACTION_TIME);
            AttributeDescription street, officeName;
            street.name = "Адрес"; street.type = Type::STRING;
            officeName.name = "Название офиса"; officeName.type = Type::STRING;
            offices.addColumn(street); offices.addColumn(officeName);
            db.createTable(abonents);
            db.createTable(offices);
            db.showTableHeader("Абоненты ТЕЛЕ-3");
            db.showTableHeader("Офисы ТЕЛЕ-3");
        }

        for (int i = 0; i < 30; i++) {
            Row r;
            r.spatialKey.name = "Позиция";
            r.temporalKey.name = "Время";
            r.spatialKey.type = SpatialType::POINT;
            r.temporalKey.type = TemporalType::VALID_TIME;
            r.spatialKey.points.push_back({(float)(((rand() / ((double)RAND_MAX)))  * 2.0 - 1.0), (float)(((rand() / ((double)RAND_MAX)))  * 2.0 - 1.0)});
            r.temporalKey.validTimeS = Date::getRandomDate(Date(2012, 01, 01, 00, 00, 00, 0), Date(2016,12, 31, 23, 59, 59, 999999));
            r.temporalKey.validTimeE.setFromCode(r.temporalKey.validTimeS.codeDate() + (rand() / ((double)RAND_MAX)) * 60 * 999999);
            if (r.temporalKey.validTimeS.codeDate() > r.temporalKey.validTimeE.codeDate()) {
                myswap(r.temporalKey.validTimeS, r.temporalKey.validTimeE);
            }
            Attribute name, secondName;
            name.setName("Имя");
            secondName.setName("Фамилия");
            name.setValue(names[((rand() / ((double)RAND_MAX))) * names.size()]);
            secondName.setValue(secondNames[((rand() / ((double)RAND_MAX))) * secondNames.size()]);
            r.addAttribute(name);
            r.addAttribute(secondName);
            db.insertRow("Абоненты ТЕЛЕ-3", r);
        }

        db.showTable("Абоненты ТЕЛЕ-3");

        result = db.selectRow("Абоненты ТЕЛЕ-3", SELECT_ALL_ROWS()).unwrap();

        for (uint i = 0; i < result.size(); i++) {
            printf("\\(%f; %f\\) & %s - %s & %s & %s\\\\ \\hline\n", result[i].spatialKey.points[0].x, result[i].spatialKey.points[0].y,
                    result[i].temporalKey.validTimeS.toString().c_str(), result[i].temporalKey.validTimeE.toString().c_str(),
                    result[i].getAttribute(0).unwrap().getString().unwrap().c_str(), result[i].getAttribute(1).unwrap().getString().unwrap().c_str());
        }
    }
    std::vector<std::vector<float2>> polygons;
    {

        polygons.resize(2);
        polygons[0].push_back({0.0 - 1.0, 0.0});
        polygons[0].push_back({1.0 - 1.0, 0.0});
        polygons[0].push_back({1.0 - 1.0, 0.8});
        polygons[0].push_back({0.7 - 1.0, 0.8});
        polygons[0].push_back({0.7 - 1.0, 0.5});
        polygons[0].push_back({0.3 - 1.0, 0.5});
        polygons[0].push_back({0.3 - 1.0, 0.8});
        polygons[0].push_back({0.0 - 1.0, 0.8});

        polygons[1].push_back({1.0, -0.1});
        polygons[1].push_back({0.0, -0.6});
        polygons[1].push_back({-1.0, -0.1});


        for (int i = 0; i < 2; i++) {
            Row r;
            r.spatialKey.name = "Граница офиса";
            r.temporalKey.name = "Время внесения в таблицу";
            r.spatialKey.type = SpatialType::POLYGON;
            r.temporalKey.type = TemporalType::TRANSACTION_TIME;
            r.spatialKey.points = polygons[i];
            Attribute street, officeName;
            street.setName("Адрес");
            officeName.setName("Название офиса");
            street.setValue("Улица \"" + names[((rand() / ((double)RAND_MAX))) * names.size()] + "\"");
            officeName.setValue("Офис имени \"" + secondNames[((rand() / ((double)RAND_MAX))) * secondNames.size()] + "\"");
            r.addAttribute(street);
            r.addAttribute(officeName);
            db.insertRow("Офисы ТЕЛЕ-3", r);
        }
        db.showTable("Офисы ТЕЛЕ-3");

        std::vector<Row> result = db.selectRow("Офисы ТЕЛЕ-3", SELECT_ALL_ROWS()).unwrap();

        for (uint i = 0; i < result.size(); i++) {
            printf("\\{");
            for (uint j = 0; j < result[i].spatialKey.points.size(); j++) {
                printf("\\(%0.1f; %0.1f\\)", result[i].spatialKey.points[j].x, result[i].spatialKey.points[j].y);
                if (j < result[i].spatialKey.points.size() - 1) {
                    printf(", ");
                }
            }
            printf("\\} & ");
            printf("%s & ", result[i].temporalKey.transactionTime.toString().c_str());
            printf("%s & ", result[i].getAttribute(0).unwrap().getString().unwrap().c_str());
            printf("%s ", result[i].getAttribute(1).unwrap().getString().unwrap().c_str());
            /*printf("(%f; %f) & %s & %s & %s", result[i].spatialKey.points[0].x, result[i].spatialKey.points[0].y,
                    result[i].temporalKey.validTimeS.toString().c_str(), result[i].temporalKey.validTimeE.toString().c_str(),
                    result[i].getAttribute(0).getString().c_str(), result[i].getAttribute(1).getString().c_str());*/
            printf("\\\\ \\hline\n");
        }
    }

    std::unique_ptr<TempTable> officesTT = db.selectTable("Офисы ТЕЛЕ-3").unwrap();
    std::unique_ptr<TempTable> abonentsTT = db.selectTable("Абоненты ТЕЛЕ-3").unwrap();
    std::unique_ptr<TempTable> abonentsTT_filtered = db.filter(abonentsTT, tester()).unwrap();
    db.showTable(abonentsTT_filtered);
    std::unique_ptr<TempTable> output = db.polygonxpointPointsInPolygon(officesTT, abonentsTT_filtered).unwrap();
    db.showTable(output);
    std::vector<Row> output_rows;
    output_rows = db.selectRow(output, SELECT_ALL_ROWS()).unwrap();
    result = db.selectRow(abonentsTT_filtered, SELECT_ALL_ROWS()).unwrap();
    db.showTable(output_rows[0].getAttribute(2).unwrap().getSet().unwrap());

    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        exit( EXIT_FAILURE );
    }
    glewExperimental = GL_TRUE;
    glewInit();

    GLFWwindow* window = glfwCreateWindow (1000, 1000, "Hello Triangle", NULL, NULL);
    if (!window) {
        fprintf (stderr, "ERROR: could not open window with GLFW3\n");
        glfwTerminate();
    }
    glfwMakeContextCurrent (window);
    reshape(1000, 1000);

    do{
        glClearColor(1, 1, 1, 1);
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        reshape(width, height);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glPointSize(12.0);
/*
        for (int i = 0; i < result.size(); i++) {
            glBegin(GL_POINTS);
                glColor3f(0, 0, 0);
                glVertex2f(result[i].spatialKey.points[0].x, result[i].spatialKey.points[0].y);
            glEnd();
        }*/

        for (uint i = 0; i < 1; i++) {
            std::vector<Row> rows;
            auto ptr = output_rows[i].getAttribute(2).unwrap().getSet().unwrap();
            rows = db.selectRow(ptr, SELECT_ALL_ROWS()).unwrap();

            for (uint j = 0; j < rows.size(); j++) {
                glBegin(GL_POINTS);
                    glColor3f(1, 0, 0);
                    glVertex2f(rows[j].spatialKey.points[0].x, rows[j].spatialKey.points[0].y);
                glEnd();
            }
        }

        for (uint i = 0; i < polygons.size(); i++) {
            glBegin(GL_LINE_LOOP);
            for (uint j = 0; j < polygons[i].size(); j++) {
                glColor3f(0, 0, 0);
                glVertex2f(polygons[i][j].x, polygons[i][j].y);
            }
            glEnd();
        }


        glfwSwapBuffers(window);
        glfwPollEvents();

    }
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose(window) == 0 );
}

void test() {
    DataBase &db = DataBase::getInstance();
    setlocale(LC_ALL, "ru_RU.utf8");

    pointline();
    db.deinit();
    exit(0);
    auto result = db.loadFromDisk("/home/sargarass/tmp/db/test.sdata");
    if (result.isErr()) {
        Error<std::string> err = result.unwrapErr();
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Was not loaded: %s %s:%d", err.what.c_str(), err.function, err.line);
        exit(-1);
    } else {
        gLogWrite(LOG_MESSAGE_TYPE::INFO, "db was loaded");
    }

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
//    inserted.push_back(newRow);

//    db.createTable(table);
//    db.insertRow("test1", newRow);
//    newRow.spatialKey.points.clear();
//    newRow.spatialKey.points.push_back(make_float2(0.5, 1.1));
//    inserted.push_back(newRow);
//    db.insertRow("test1", newRow);
//    newRow.spatialKey.points.clear();
//    newRow.spatialKey.points.push_back(make_float2(0.755, 0.0));
//    inserted.push_back(newRow);
//    db.insertRow("test1", newRow);
//    newRow.spatialKey.points.clear();
//    newRow.spatialKey.points.push_back(make_float2(0.42, 0.0));
//    inserted.push_back(newRow);
//    db.insertRow("test1", newRow);
//    newRow.spatialKey.points.clear();
//    newRow.spatialKey.points.push_back(make_float2(0.1241, 0.0));
//    inserted.push_back(newRow);
//    db.insertRow("test1", newRow);
//    newRow.spatialKey.points.clear();
//    newRow.spatialKey.points.push_back(make_float2(0.54, 0.0));
//    inserted.push_back(newRow);
//    db.insertRow("test1", newRow);


//    std::vector<float2> random;
//    for (int i = 0; i < 30; i++) {
//        Date d1, d2;
//        d1 = Date::getRandomDate();
//        d2 = Date::getRandomDate();
//        if (d1.codeDate() > d2.codeDate()) {
//            std::swap(d1, d2);
//        }

//        newRow.temporalKey.validTimeS = d1;
//        newRow.temporalKey.validTimeE = d2;
//        newRow.spatialKey.points.clear();
//        newRow.spatialKey.points.push_back(make_float2(rand() / float(RAND_MAX) * 10, rand() / float(RAND_MAX) * 10));
//        random.push_back(newRow.spatialKey.points[0]);
//        //db.insertRow("test1", newRow);
//        inserted.push_back(newRow);
//    }
//    db.insertRow("test1", inserted);


//    table.setName("test2");
//    table.setSpatialKey("Полигонъ", SpatialType::POLYGON);
//    db.createTable(table);

//    newRow.spatialKey.points.clear();
//    newRow.spatialKey.points.push_back(make_float2(0.0, -1.0));
//    newRow.spatialKey.points.push_back(make_float2(1.0, 0.0));
//    newRow.spatialKey.points.push_back(make_float2(0.0, 10));
//    newRow.spatialKey.points.push_back(make_float2(-1.0, 0.0));
//    newRow.spatialKey.name = "Полигонъ";
//    newRow.spatialKey.type = SpatialType::POLYGON;
//    db.insertRow("test2", newRow);

//    //db.insertRow("test2", newRow);
//    db.showTable("test1");
//    std::set<Attribute> atrSet;
//    atr2.setValue("MOSCOW NEVER CRY");
//    atr.setValue(4542.5489);
//    atrSet.insert(atr2);
//    atrSet.insert(atr);


//    Predicate p = tester();
//    //db.showTable("test1");
///*
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
    db.showTableHeader("test3");

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

    std::unique_ptr<TempTable> temptable1 = db.selectTable("test1").unwrap();
    std::unique_ptr<TempTable> temptable2 = db.selectTable("test2").unwrap();
    std::unique_ptr<TempTable> temptable3 = db.selectTable("test3").unwrap();

    std::vector<Row> selected = db.selectRow(temptable1, SELECT_ALL_ROWS()).unwrap();

//    for (int i = 0; i < selected.size(); i++) {
//        std::cout << selected[i].temporalKey.validTimeS.toString() << " " << selected[i].temporalKey.validTimeE.toString() << std::endl;
//        std::cout << selected[i].getAttributeSize() << std::endl;

//        for (int j = 0; j < selected[i].getAttributeSize(); j++) {
//            std::cout << selected[i].getAttribute(j).getName() << " : "
//            << typeToString(selected[i].getAttribute(j).getType()) << " : ";
//            if (!selected[i].getAttribute(j).isNull()) {
//                switch (selected[i].getAttribute(j).getType()) {
//                    case Type::REAL:
//                        std::cout << selected[i].getAttribute(j).getReal().unwrap() << " ";
//                        break;
//                    case Type::STRING:
//                        std::cout << selected[i].getAttribute(j).getString().unwrap() << " ";
//                        break;
//                    case Type::INT:
//                        std::cout << selected[i].getAttribute(j).getInt().unwrap() << " ";
//                        break;
//                }
//            } else {
//                std::cout << "NULL" << std::endl;
//            }
//        }
//        std::cout << std::endl;
//        std::cout << std::endl;
//    }
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
  // db.selectTable("test2", temptable2);
  // db.selectTable("test3", temptable3);
    db.showTable(temptable1);
    db.showTable(temptable2);
    db.showTable(temptable3);
    Result<std::unique_ptr<TempTable>, Error<std::string>> res;
    res = db.pointxpointKnearestNeighbor(temptable1, temptable1, 24);
    if (res.isErr()) {
        Error<std::string> err = res.unwrapErr();
        std::cout <<err.what << std::endl;
        std::cout <<err.line << std::endl;
        std::cout <<err.file << std::endl;
        std::cout <<err.function << std::endl;

        exit(-1);
    }
    std::unique_ptr<TempTable> temptable5 = res.unwrap();
    db.showTable(temptable5);

    std::unique_ptr<TempTable> temptable4 = db.linexpointPointsInBufferLine(temptable3, temptable1, 1.0f).unwrap();
    db.showTable(temptable4);

    std::unique_ptr<TempTable> temptable6 = db.polygonxpointPointsInPolygon(temptable2, temptable1).unwrap();
    db.showTable(temptable6);

    std::unique_ptr<TempTable> temptable7 = db.polygonxpointPointsInPolygon(temptable6, temptable1).unwrap();
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
//        db.insertRow("test1", inserted).unwrap();
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
    auto saveondiskres = db.saveOnDisk("/home/sargarass/tmp/db/test.sdata");
    if (saveondiskres.isErr()) {
        Error<std::string> err = saveondiskres.unwrapErr();
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Was not loaded: %s %s:%d", err.what.c_str(), err.function, err.line);
        exit(-1);
    } else {
        gLogWrite(LOG_MESSAGE_TYPE::INFO, "db was saved");
    }
    //std::cout << db.loadFromDisk("/home/sargarass/tmp/db/test.sdata") << std::endl;
}
