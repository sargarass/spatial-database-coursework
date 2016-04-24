#pragma once
#include <list>
#include <algorithm>
#include <stack>
#include <map>
#include <vector>
class SingletonFactory;

class Singleton {
    friend class SingletonFactory;
public:
    Singleton& operator=(Singleton const &) = delete;
    Singleton(Singleton &singleton) = delete;

    bool dependOn(Singleton &s);
protected:
    Singleton() {id = 0xFFFFFFFFFFFULL;}
    bool checkOnCicle(Singleton *link);
    virtual ~Singleton(){}
    std::list<Singleton *> linkFrom;
    std::list<Singleton *> linkTo;
    void delLinkTo(Singleton *link);
    uint64_t id;
};

class SingletonFactory {
public:
    static SingletonFactory &getInstance() {
        static SingletonFactory s;
        return s;
    }

    template<typename T>
    bool registration(T *singleton) {
        Singleton *singleton_ptr = dynamic_cast<Singleton*>(singleton);
        if (singleton_ptr == nullptr) {
            return false;
        }
        singleton_ptr->id = singletonGraph.size();
        singletonGraph.push_back(singleton_ptr);
        return true;
    }

    virtual ~SingletonFactory();
private:
    void bfsDelete(std::vector<bool> &used, Singleton *link);
    std::vector<Singleton *> singletonGraph;
};
