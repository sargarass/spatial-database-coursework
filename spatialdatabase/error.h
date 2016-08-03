#pragma once
#include <memory>

template<typename E>
struct Error {
    Error(E&& what, char const *function, char const *file, uint line) :
        what { what },
        function { function },
        line { line },
        file { file }
    {}

    Error(Error const &err) :
        what { err.what },
        line { err.line },
        file { err.file },
        function { err.function }
    {}

    Error(Error &&err) :
        what { std::move(err.what) },
        line { std::move(err.line) },
        function { std::move(err.function) },
        file { std::move(err.file) }
    {}

    E what;
    uint line;
    char const *file;
    char const *function;
};

#define ERROR_ARGS(x) x, __PRETTY_FUNCTION__, __FILE__, __LINE__
