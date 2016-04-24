#include "singletonmanager.h"
#include <queue>
#include "log.h"
bool Singleton::dependOn(Singleton &s) {
    if (&s == this) {
        return false;
    }

    linkFrom.push_back(&s);
    s.linkTo.push_back(this);
    if (checkOnCicle(this)) {
        linkFrom.erase(std::find(linkFrom.begin(), linkFrom.end(), &s));
        s.linkTo.erase(std::find(s.linkTo.begin(), s.linkTo.end(), this));
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "Cicle is found!!!");
        exit(-1);
        return false;
    }
    return true;
}

bool Singleton::checkOnCicle(Singleton *link){
   std::stack<Singleton *> dfs;
   std::map<uint64_t, bool> used;

   dfs.push(link);
   Singleton *root = link;
   while(!dfs.empty()) {
       link = dfs.top();
       used[link->id] = true;
       dfs.pop();

       for (auto &v : link->linkFrom) {
           auto it = used.find(v->id);
           if (v->id == root->id) {
               printf("Circle %d -> %d", v->id, root->id);
               return true;
           }

           if (it == used.end()) {
               dfs.push(v);
           }
       }
   }
   return false;
}

void Singleton::delLinkTo(Singleton *link) {
    auto it = std::find(linkTo.begin(), linkTo.end(), link);
    if (it != linkTo.end()) {
        linkTo.erase(it);
    }
}

void SingletonFactory::bfsDelete(std::vector<bool> &used, Singleton *link) {
    std::queue<Singleton *> bfs;
    std::stack<Singleton *> stack;
    std::vector<bool> used2;
    used2 = used;

    bfs.push(link);
    while(!bfs.empty()) {
        link = bfs.front();
        bfs.pop();
        stack.push(link);
        for (auto &v : link->linkTo) {
            if (!used2[v->id]) {
                bfs.push(v);
                used2[v->id] = true;
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
            bfsDelete(erased, singletonGraph[i]);
        }
    }
    fflush(stdout);
}
