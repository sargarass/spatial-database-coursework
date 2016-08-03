#ifndef RESULT_H
#define RESULT_H
#include <algorithm>
#include <iostream>
#include "result_impl.h"

enum class ResultType {
    Ok,
    Err
};

template<typename T, typename CleanT = typename std::decay<T>::type>
types::Ok<CleanT> Ok(T &&val) {
    return types::Ok<CleanT>(std::forward<T>(val));
}

template<typename CleanType = void>
types::Ok<void> Ok() {
    return types::Ok<void>();
}

template<typename E, typename CleanE = typename std::decay<E>::type>
inline types::Err<CleanE> Err(E&& val) {
    return types::Err<CleanE>(std::forward<E>(val));
}

template<typename T, typename E>
class Result {
public:
    static_assert(!std::is_same<E, void>::value, "void error type is not allowed");
    typedef result_impl::Storage<T, E> storage_t;
    Result()
        : res {ResultType::Ok}
    {}

    Result(types::Ok<T> &&ok)
        : res {ResultType::Ok}
    {
        storage.constructor(std::forward<types::Ok<T>>(ok));
    }

    Result(types::Err<E> &&err)
        : res {ResultType::Err}
    {
        storage.constructor(std::forward<types::Err<E>>(err));
    }

    Result(Result const &other) {
        res = other.res;
        if (isOk()) {
            result_impl::Constructor<T, E>::copy(storage, other.storage, result_impl::ok_tag());
        } else {
            result_impl::Constructor<T, E>::copy(storage, other.storage, result_impl::err_tag());
        }
    }

    Result(Result &&other) {
        res = other.res;
        if (isOk()) {
            result_impl::Constructor<T, E>::move(storage, other.storage, result_impl::ok_tag());
        } else {
            result_impl::Constructor<T, E>::move(storage, other.storage, result_impl::err_tag());
        }
    }

    inline void cleanStorage() {
        if (isOk()) {
            storage.destroy(result_impl::ok_tag());
        } else {
            storage.destroy(result_impl::err_tag());
        }
    }

    inline bool isErr() const {
        return (res == ResultType::Err);
    }

    inline bool isOk() const {
        return (res == ResultType::Ok);
    }

    ~Result() {
        cleanStorage();
    }

    template<typename U = T> inline
    typename std::enable_if<!std::is_same<U, void>::value, U&&>::type
    expect(std::string const &v) {
        if (isErr()) {
            std::cerr << v << std::endl;
            std::terminate();
        }
        return std::move(storage.template get<T>());
    }
    template<typename U = T> inline
    typename std::enable_if<!std::is_same<U, void>::value, U&&>::type
    unwrap() {
        return expect("Attempting to unwrap an error Result");
    }

    //// T = void
    template<typename U = T> inline
    typename std::enable_if<std::is_same<U, void>::value, U>::type
    expect(std::string const &v) {
        if (isErr()) {
            std::cerr << v << std::endl;
            std::terminate();
        }
    }

    template<typename U = T>
    typename std::enable_if<std::is_same<U, void>::value, U>::type
    unwrap() {
       expect("Attempting to unwrap an error Result");
    }
    ////

    E&& unwrapErr() {
        if (isErr()) {
            return std::move(storage.template get<E>());
        }
        std::cerr << "Attempting to unwrap an ok Result" << std::endl;
        std::terminate();
    }

public:
    storage_t storage;
    ResultType res;
};

#define TRY(...) \
({ \
    auto res = std::move(__VA_ARGS__); \
    \
    if (res.isErr()) { \
        typedef typename result_impl::ResultErrType<decltype(res)>::type E; \
        return types::Err<E>(res.unwrapErr()); \
    } \
    res.unwrap(); \
});

std::string to_string(ResultType type);
#endif // RESULT_H
