#ifndef DEBUG_OPENMP_H
#define DEBUG_OPENMP_H

#include <string>

#ifdef NDEBUG

#define DEBUG_ENABLE(X)
#define DebugOpenMPInit()

#else

extern bool DebugOpenMPFlag;
void DebugOpenMPInit();

#define DEBUG_ENABLE(X)                                                        \
    do {                                                                       \
        if (DebugOpenMPFlag) {                                                 \
            X;                                                                 \
        }                                                                      \
    } while (false)

#endif

[[noreturn]] void fatalError(const std::string &msg, const char *file, int line);
#define FATAL_ERROR(msg) fatalError(msg, __FILE__, __LINE__)

#endif
