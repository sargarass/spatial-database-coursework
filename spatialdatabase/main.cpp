#include <string>
#include <set>
#include <map>
#include <iostream>
#include "timer.h"
#include "database.h"
#include "consolewriter.h"
#include "testrequests.h"
/*
 * выборка строк с базы на cpu+ (доработать temptable)
 * сохранение на диск с загрузкой +
 * доделать полигон х поинт +-
 * вставка кучи строк +
 * если и делать операцию над линией то ближайшие точки к ней. +-
 */

int main()
{
    //srand(time(0));
    DataBase &db = DataBase::getInstance();
    Log::getInstance().showFilePathLevel(2);
    ConsoleWriter::getInstance().showDebug(false);

    test();
    db.deinit();
    return 0;
}
