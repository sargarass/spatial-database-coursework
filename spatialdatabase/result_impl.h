#ifndef RESULT_IMPL_H
#define RESULT_IMPL_H
#include <iostream>
#include <stdlib.h>
//#define RESULT_IMPL_DEBUG_OUTPUT
#ifdef RESULT_IMPL_DEBUG_OUTPUT
#define RI_DEBUG_OUTPUT(x) x
#else
#define RI_DEBUG_OUTPUT(x)
#endif

static inline void UNUSED_PARAM_HANDLER_RESULT(){}
template <typename Head, typename ...Tail>
static inline void UNUSED_PARAM_HANDLER_RESULT(Head car, Tail ...cdr) { ((void) car); UNUSED_PARAM_HANDLER_RESULT(cdr...);}

template<typename T, typename E>
class Result;

namespace types {
    template<typename T>
    struct Ok {
        Ok(const T &val)
            : val(val)
        {
            RI_DEBUG_OUTPUT(std::printf("%p: Ok constructor\n", this));
        }

        Ok(Ok<T> &&ok) {
            std::swap(this->val, ok.val);
        }

        template<typename U = T>
        Ok(U &&val)
            : val(std::forward<U>(val))
        {
            RI_DEBUG_OUTPUT(std::printf("%p: Ok move constructor\n", this));
        }

        ~Ok() {
            RI_DEBUG_OUTPUT(std::printf("%p: Ok destructor\n", this));
        }
        T val;
    };

    template<> struct Ok<void> {};

    template<typename E>
    struct Err {
        Err(const E& val)
            : val(val)
        {
            RI_DEBUG_OUTPUT(std::printf("%p: Err constructor\n", this));
        }

        Err(E&& val) : val(std::move(val)) {
            RI_DEBUG_OUTPUT(std::printf("%p: Err move constructor\n", this));
        }

        ~Err() {
            RI_DEBUG_OUTPUT(std::printf("%p: Err destructor\n", this));
        }
        E val;
    };
}

namespace result_impl {
    template<typename R>
    struct ResultErrType { typedef R type; };

    template<typename R>
    struct ResultOkType { typedef R type; };

    template<typename T, typename E>
    struct ResultErrType<Result<T, E>> {
        typedef typename std::remove_reference<E>::type type;
    };

    template<typename T, typename E>
    struct ResultOkType<Result<T, E>> {
        typedef typename std::remove_reference<T>::type type;
    };

    struct ok_tag {};
    struct err_tag {};

    template<typename T, typename E>
    struct Storage {
        static constexpr size_t size = (sizeof(T) > sizeof(E))? sizeof(T) : sizeof(E);
        static constexpr size_t align = (sizeof(T) > sizeof(E))? alignof(T) : alignof(E);
        typedef typename std::aligned_storage<size, align>::type storage_t;
        Storage()
            : init(false)
        {}

        void constructor(types::Ok<T> const &ok) {
            new (&storage) T(ok.val);
            init = true;
            RI_DEBUG_OUTPUT(std::printf("%p: storage ok constuctor\n", this));
        }

        void constructor(types::Err<E> const &err) {
            new (&storage) E(err.val);
            init = true;
            RI_DEBUG_OUTPUT(std::printf("%p: storage err constuctor\n", this));
        }

        void constructor(types::Ok<T> &&ok) {
            typedef typename std::decay<T>::type CleanT;
            new (&storage) CleanT(std::move(ok.val));
            init = true;
            RI_DEBUG_OUTPUT(std::printf("%p: storage ok move constuctor\n", this));
        }

        void constructor(types::Err<E> &&err) {
            typedef typename std::decay<E>::type CleanErr;
            new (&storage) CleanErr(err.val);
            init = true;
            RI_DEBUG_OUTPUT(std::printf("%p: storage err move constuctor\n", this));
        }

        template<typename U>
        void constructorRaw(U &&val) {
            typedef typename std::decay<U>::type CleanU;
            new (&storage) CleanU(std::forward<U>(val));
            init = true;
            RI_DEBUG_OUTPUT(std::printf("%p: storage raw constuctor\n", this));
        }

        template<typename U>
        const U &get() const {
            return *reinterpret_cast<const U*>(&storage);
        }

        template<typename U>
        U &get() {
            return *reinterpret_cast<U*>(&storage);
        }

        void destroy(result_impl::ok_tag) {
            if (init) {
                RI_DEBUG_OUTPUT(std::printf("%p: storage ok destuctor\n", this));
                get<T>().~T();
                init = false;
            }
        }

        void destroy(result_impl::err_tag) {
            if (init) {
                RI_DEBUG_OUTPUT(std::printf("%p: storage err destuctor\n", this));
                get<E>().~E();
                init = false;
            }
        }

        storage_t storage;
        bool init;
    };

    template<typename E>
    struct Storage<void, E> {
        typedef typename std::aligned_storage<sizeof(E), alignof(E)>::type storage_t;
        Storage()
            : init(false)
        {}

        void constructor(types::Ok<void> const &ok) {
            init = true;
            RI_DEBUG_OUTPUT(std::printf("%p: storage ok constuctor\n", this));
        }

        void constructor(types::Err<E> const &err) {
            new (&storage) E(err.val);
            init = true;
            RI_DEBUG_OUTPUT(std::printf("%p: storage err constuctor\n", this));
        }

        void constructor(types::Ok<void> &&ok) {
            UNUSED_PARAM_HANDLER_RESULT(ok);
            init = true;
            RI_DEBUG_OUTPUT(std::printf("%p: storage ok move constuctor\n", this));
        }

        void constructor(types::Err<E> &&err) {
            typedef typename std::decay<E>::type CleanErr;
            new (&storage) CleanErr(err.val);
            init = true;
            RI_DEBUG_OUTPUT(std::printf("%p: storage err move constuctor\n", this));
        }

        template<typename U>
        void constructorRaw(U &&val) {
            typedef typename std::decay<U>::type CleanU;
            new (&storage) CleanU(std::forward<U>(val));
            init = true;
            RI_DEBUG_OUTPUT(std::printf("%p: storage raw constuctor\n", this));
        }

        template<typename U>
        const U &get() const {
            return *reinterpret_cast<const U*>(&storage);
        }

        template<typename U>
        U &get() {
            return *reinterpret_cast<U*>(&storage);
        }

        void destroy(result_impl::ok_tag) {
            if (init) {
                RI_DEBUG_OUTPUT(std::printf("%p: storage ok destuctor\n", this));
            }
            init = false;
        }

        void destroy(result_impl::err_tag) {
            if (init) {
                RI_DEBUG_OUTPUT(std::printf("%p: storage err destuctor\n", this));
                get<E>().~E();
                init = false;
            }
        }

        storage_t storage;
        bool init;
    };

    template<typename T, typename E>
    struct Constructor {
        typedef Storage<T, E> storage_t;
        static void move(storage_t &dst, storage_t &src, ok_tag) {
            RI_DEBUG_OUTPUT(std::printf("%p to %p: storage ok move\n", &src, &dst));
            dst.constructorRaw(std::move(src.template get<T>()));
            src.destroy(ok_tag());
        }

        static void move(storage_t &dst, storage_t &src, err_tag) {
            RI_DEBUG_OUTPUT(std::printf("%p to %p: storage err move\n", &src, &dst));
            dst.constructorRaw(std::move(src.template get<E>()));
            src.destroy(err_tag());
        }

        static void copy(storage_t &dst, storage_t const &src, ok_tag) {
            RI_DEBUG_OUTPUT(std::printf("%p to %p: storage ok copy\n", &src, &dst));
            dst.constructorRaw(src.template get<T>());
        }

        static void copy(storage_t &dst, storage_t const &src, err_tag) {
            RI_DEBUG_OUTPUT(std::printf("%p to %p: storage err copy\n", &src, &dst));
            dst.constructorRaw(src.template get<E>());
        }
    };

    template<typename E>
    struct Constructor<void, E> {
        typedef Storage<void, E> storage_t;
        static void move(storage_t &dst, storage_t &src, ok_tag) { UNUSED_PARAM_HANDLER_RESULT(dst, src); }

        static void move(storage_t &dst, storage_t &src, err_tag) {
            RI_DEBUG_OUTPUT(std::printf("%p to %p: storage err move\n", &src, &dst));
            dst.constructorRaw(std::move(src.template get<E>()));
            src.destroy(err_tag());
        }

        static void copy(storage_t &dst, storage_t const &src, ok_tag) { UNUSED_PARAM_HANDLER_RESULT(dst, src); }

        static void copy(storage_t &dst, storage_t const &src, err_tag) {
            RI_DEBUG_OUTPUT(std::printf("%p to %p: storage err copy\n", &src, &dst));
            dst.constructorRaw(src.template get<E>());
        }
    };
}

#endif // RESULT_IMPL_H
