#include "common.cuh"

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

namespace lsso_equilibrium {
namespace {

constexpr int64_t kNativeContractVersion = 10;

bool requires_grad(const c10::optional<at::Tensor>& value) {
    return value.has_value() && value->requires_grad();
}

at::Tensor forward_inference_autograd(
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const c10::optional<at::Tensor>& valid_counts) {
    const bool input_requires_grad =
        projected.requires_grad() || core_base_raw.requires_grad() ||
        core_drive_weight.requires_grad() || eta_raw.requires_grad() ||
        requires_grad(valid_counts);
    TORCH_CHECK(
        !at::GradMode::is_enabled() || !input_requires_grad,
        "lsso_equilibrium::forward_inference is an inference-only entry point "
        "and cannot participate in autograd directly; use lsso.ball.cuda.fast_mix()"
    );

    at::AutoDispatchBelowAutograd guard;
    return forward_inference_cuda(
        projected,
        core_base_raw,
        core_drive_weight,
        eta_raw,
        valid_counts
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> forward_train_autograd(
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const c10::optional<at::Tensor>& valid_counts) {
    const bool input_requires_grad =
        projected.requires_grad() || core_base_raw.requires_grad() ||
        core_drive_weight.requires_grad() || eta_raw.requires_grad() ||
        requires_grad(valid_counts);
    TORCH_CHECK(
        !at::GradMode::is_enabled() || !input_requires_grad,
        "lsso_equilibrium::forward_train is a private tape-producing entry point "
        "and cannot participate in autograd directly; use lsso.ball.cuda.fast_mix()"
    );

    at::AutoDispatchBelowAutograd guard;
    return forward_train_cuda(
        projected,
        core_base_raw,
        core_drive_weight,
        eta_raw,
        valid_counts
    );
}

int64_t contract_version() {
    return kNativeContractVersion;
}

}  // namespace
}  // namespace lsso_equilibrium

TORCH_LIBRARY(lsso_equilibrium, module) {
    module.def(
        "contract_version() -> int",
        TORCH_FN(lsso_equilibrium::contract_version)
    );
    module.def(
        "forward_inference(Tensor projected, Tensor core_base_raw, Tensor core_drive_weight, "
        "Tensor eta_raw, Tensor? valid_counts=None) -> Tensor");
    module.def(
        "forward_train(Tensor projected, Tensor core_base_raw, Tensor core_drive_weight, "
        "Tensor eta_raw, Tensor? valid_counts=None) "
        "-> (Tensor, Tensor, Tensor)");
    module.def(
        "backward(Tensor grad_output, Tensor projected, Tensor core_base_raw, "
        "Tensor core_drive_weight, Tensor eta_raw, Tensor tape, Tensor pivots, "
        "Tensor? valid_counts=None) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(lsso_equilibrium, CUDA, module) {
    module.impl("forward_inference", TORCH_FN(lsso_equilibrium::forward_inference_cuda));
    module.impl("forward_train", TORCH_FN(lsso_equilibrium::forward_train_cuda));
    module.impl("backward", TORCH_FN(lsso_equilibrium::backward_cuda));
}

TORCH_LIBRARY_IMPL(lsso_equilibrium, Autograd, module) {
    module.impl("forward_inference", TORCH_FN(lsso_equilibrium::forward_inference_autograd));
    module.impl("forward_train", TORCH_FN(lsso_equilibrium::forward_train_autograd));
}
