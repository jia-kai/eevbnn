#include <cstdint>

static inline int32_t atomic_fetch_add_i32(int32_t* addr, int32_t delta) {
    return __atomic_fetch_add(addr, delta, __ATOMIC_SEQ_CST);
}

static inline int32_t atomic_load_i32(const int32_t* addr) {
    int32_t ret;
    __atomic_load(addr, &ret, __ATOMIC_SEQ_CST);
    return ret;
}

