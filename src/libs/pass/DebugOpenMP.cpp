#include <cstdlib>
#include <iostream>
#include <string>

bool DebugOpenMPFlag;
void DebugOpenMPInit() {
    char *DebugStr = getenv("NUMBA_DEBUG_OPENMP_LLVM_PASS");
    DebugOpenMPFlag = false;
    if(DebugStr)
        DebugOpenMPFlag = (std::stoi(DebugStr) >= 1);
}

[[noreturn]] void fatalError(const std::string &msg, const char *file, int line) {
    std::cerr << "Fatal error @ " << file << ":" << line << " :: " << msg << "\n";
    std::abort();
}
