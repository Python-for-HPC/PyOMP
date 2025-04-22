extern void NRT_MemSys_init();

__attribute__((constructor)) static void PyOMP_NRT_Init() { NRT_MemSys_init(); }
