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
    // тесты
    //pointline();
    tele3();
    db.deinit();
    exit(0);
}
