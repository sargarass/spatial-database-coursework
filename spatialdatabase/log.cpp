#include "log.h"
#include "consolewriter.h"
#include <stdarg.h>
#include "string.h"
#include <string>
Log::Log() {
    m_fileLVL = -1;
}

Log::~Log() {
    gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "delete Log");
}

void Log::showFilePathLevel(int lvl) {
    m_fileLVL = lvl;
}

void Log::write(LOG_MESSAGE_TYPE type, std::string const &fileName, std::string const &functionName, int64_t const line, const char *format, ...) {
    Message msg = {"" , type};

    std::string tmp;
    tmp.resize(256);

    va_list arglist;
    va_start( arglist, format );
    vsnprintf(&tmp[0], 255, format, arglist);
    va_end( arglist );


    if (fileName != "") {
        if (m_fileLVL == -1) {
            msg.text += fileName;
        } else {
            int v = m_fileLVL;
            int i;
            for (i = fileName.length(); i >= 0 && v > 0; --i) {
                if (fileName[i] == '/') {
                    v--;
                }
            }

            if (v == 0) {
                msg.text.append(fileName.begin() + i + 2, fileName.end());
            } else {
                msg.text += fileName;
            }
        }
        msg.text += ":";
    }

    if (line >= 0) {
        msg.text += std::to_string(line) + ":";
    }


    if (functionName != "") {
        msg.text += functionName + ": ";
    }

    msg.text += tmp;
    push(msg);
}

void Log::push(Message &msg) {
    m_queue.push_back(msg);

    if (m_queue.size() > 5) {
        m_queue.pop_front();
    }

    for (auto subscriber : m_subscribers) {
        subscriber->notify(msg);
    }
}

void Log::subscribe(ILogSubscriber *subscriber) {
    m_subscribers.push_back(subscriber);
}

Log &Log::getInstance()  {
    static Log *gLog = new Log();
    static bool init = false;
    if (init == false) {
        init = true;
        SingletonFactory::getInstance().registration<Log>(gLog);
        gLog->subscribe(&ConsoleWriter::getInstance());
        dynamic_cast<Singleton*>(gLog)->dependOn(ConsoleWriter::getInstance());
    }
    return *gLog;
}
