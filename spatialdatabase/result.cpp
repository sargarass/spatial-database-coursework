#include "result.h"

std::string to_string(ResultType type) {
   switch (type) {
       case ResultType::Ok:
           return "Ok";
       case ResultType::Err:
           return "Err";
       default:
           return "Unknown";
   }
}
