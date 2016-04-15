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
    bool checkOnCicle(Singleton *link);
    Singleton() {}
    virtual ~Singleton(){}
    std::list<Singleton *> linkFrom;
    std::list<Singleton *> linkTo;
    void delLinkTo(Singleton *link);
    uint64_t id;
};

class SingletonFactory : public Singleton {
public:
    static SingletonFactory &getInstance() {
        static SingletonFactory s;
        return s;
    }

    template<typename T>
    T& create() {
        T* singleton = new T();
        Singleton* singleton_ptr = dynamic_cast<Singleton*>(singleton);
        singleton_ptr->id = singletonGraph.size();
        singletonGraph.push_back(dynamic_cast<Singleton*>(singleton_ptr));
        return *singleton;
    }

    virtual ~SingletonFactory();
private:
    void bfsDeleteFromBottomToTop(std::vector<bool> &used, Singleton *link);
    std::vector<Singleton *> singletonGraph;
};
