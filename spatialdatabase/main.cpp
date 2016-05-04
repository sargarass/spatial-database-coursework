#include <string>
#include <set>
#include <map>
#include <iostream>
#include "timer.h"
#include "database.h"
#include "consolewriter.h"
#include "testrequests.h"

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
