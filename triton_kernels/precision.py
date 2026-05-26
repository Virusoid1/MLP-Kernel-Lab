"""
全局精度配置：控制 Triton kernel 的 TF32 行为。

用法：
    from triton_kernels.precision import precision
    precision.allow_tf32 = False  # FP32 strict
"""


class _PrecisionConfig:
    def __init__(self):
        self._allow_tf32 = True

    @property
    def allow_tf32(self) -> bool:
        return self._allow_tf32

    @allow_tf32.setter
    def allow_tf32(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"allow_tf32 must be bool, got {type(value)}")
        self._allow_tf32 = value


precision = _PrecisionConfig()


def set_precision(mode: str) -> None:
    if mode == "tf32":
        precision.allow_tf32 = True
    elif mode == "fp32":
        precision.allow_tf32 = False
    else:
        raise ValueError(f"Unknown precision mode: {mode!r}. Use 'tf32' or 'fp32'.")
