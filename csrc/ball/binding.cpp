#include "common.cuh"
#include <torch/library.h>

namespace memsolve_equilibrium {
int64_t contract_version() { return 12; }
}
TORCH_LIBRARY(memsolve_equilibrium, module) {
    module.def("contract_version() -> int", TORCH_FN(memsolve_equilibrium::contract_version));
    module.def("compact_lu(Tensor a) -> (Tensor, Tensor, Tensor)");
    module.def("compact_getrs(Tensor lu, Tensor pivots, Tensor rhs, bool transpose=False) -> Tensor");
}
TORCH_LIBRARY_IMPL(memsolve_equilibrium, CUDA, module) {
    module.impl("compact_lu", TORCH_FN(memsolve_equilibrium::compact_lu_cuda));
    module.impl("compact_getrs", TORCH_FN(memsolve_equilibrium::compact_getrs_cuda));
}
TORCH_LIBRARY_IMPL(memsolve_equilibrium, Autograd, module) {
    module.impl("compact_lu", TORCH_FN(memsolve_equilibrium::compact_lu_cuda));
    module.impl("compact_getrs", TORCH_FN(memsolve_equilibrium::compact_getrs_cuda));
}
