#include "singletonmanager.h"
#include <queue>

bool Singleton::dependOn(Singleton &s) {
    if (&s == this) {
        return false;
    }

    linkFrom.push_back(&s);
    s.linkTo.push_back(this);
    if (checkOnCicle(this)) {
        linkFrom.erase(std::find(linkFrom.begin(), linkFrom.end(), &s));
        s.linkTo.erase(std::find(s.linkTo.begin(), s.linkTo.end(), this));
        printf("Cicle is found!!!\n");
        fflush(stdout);
        exit(-1);
        return false;
    }
    return true;
}

bool Singleton::checkOnCicle(Singleton *link){
   std::stack<Singleton *> dfs;
   std::map<Singleton*, bool> used;

   dfs.push(link);

   while(!dfs.empty()) {
       link = dfs.top();
       used[link] = true;
       dfs.pop();

       for (auto v : link->linkFrom) {
           if (used.find(v) != used.end()) {
               return true;
           } else {
               dfs.push(v);
           }
       }
   }
   return false;
}

void Singleton::delLinkTo(Singleton *link) {
    linkTo.erase(std::find(linkTo.begin(), linkTo.end(), link));
}

void SingletonFactory::bfsDeleteFromBottomToTop(std::vector<bool> &used, Singleton *link) {
    std::queue<Singleton *> queue;
    std::stack<Singleton *> stack;

    queue.push(link);
    while(!queue.empty()) {
        link = queue.front();
        queue.pop();
        stack.push(link);
        for (auto &v : linkTo) {
            if (!used[link->id]) {
                queue.push(v);
            }
        }
    }

    while(!stack.empty()) {
        link = stack.top();
        stack.pop();

        if (link->linkTo.empty()) {
            for (auto& linkF : link->linkFrom) {
                linkF->delLinkTo(link);
            }
            delete singletonGraph[link->id];
            singletonGraph[link->id] = nullptr;
            used[link->id] = true;
        }
    }
}

SingletonFactory::~SingletonFactory() {
    uint64_t size = singletonGraph.size();
    std::vector<bool> erased(size);
    std::fill(erased.begin(), erased.end(), 0);
    for (size_t i = 0; i < size; i++) {
        if (erased[i] == false) {
            bfsDeleteFromBottomToTop(erased, singletonGraph[i]);
        }
    }
    fflush(stdout);
}
