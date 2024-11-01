from dataclasses import dataclass


@dataclass
class Arguments:
    kernel_name: str = ""
    path: str = ""
    signature: str = ""
    arg: str = ""
    out_path: str = ""
    out_name: str = ""
    num_warps: int = ""
    num_stages: int = ""
    num_cta: int = 1
    flush_denorm: bool = False
    enable_sp3: bool = True
    global_prefetch: int = 0


@dataclass
class ShaderInfo:
    num_vgprs: int = 0
    num_sgprs: int = 0
    lds_bytes: int = 0
    sp3filename: str = ""
    cluster_dim: tuple[int, int, int] = (1, 1, 1)
    use_scratch: bool = False


@dataclass
class FFMConfig:
    variance: float = 0.01
    id: str = ""
    sp3: str = ""
    regIni: str = ""
    surfaceIni: str = ""
    name: str = ""
