#pragma once
#include "string.h"
#include "deque"
#include "string"
#include "list"
#include "singletonmanager.h"
#define SourcePos() (char const*)(SourcePosition(__FILE__, __LINE__))

class SourcePosition {
public:
    SourcePosition(std::string file, int line) {
        str = file + ":" + std::to_string(line);
    }

    std::string str;

    char const *c_str() {
        return str.c_str();
    }
    operator char const *() {
        return str.c_str();
    }
};

enum class LOG_MESSAGE_TYPE {
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    ASSERT
};

struct Message {
    std::string text;
    LOG_MESSAGE_TYPE type;
};

class ILogSubscriber {
public:
    ILogSubscriber(){}
    virtual ~ILogSubscriber(){}
    virtual void notify(Message const& msg) = 0;
};

class Log : public Singleton {
public:
    void write(LOG_MESSAGE_TYPE type, const std::string &className, const std::string &functionName, const char *format, ...);
    void subscribe(ILogSubscriber *subscriber);
    static Log &getInstance();

private:
    void push(Message& msg);
    std::list<ILogSubscriber*> m_subscribers;
    std::deque<Message> m_queue;
    virtual ~Log(){printf("delete Log();\n");}
};

extern Log gLog;
