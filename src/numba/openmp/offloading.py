import re
from .omp_runtime import (
    omp_get_num_devices,
    omp_get_default_device,
    omp_get_initial_device,
)

_device_info_output = {}
_device_info_map = {}


def add_device_info(device_id, info_text):
    """
    Add the raw device info text for a given device ID, and parse it to extract
    structured info about the device type, vendor, and architecture.

    :param device_id: Device ID as returned by OpenMP runtime
    :param info_text: Raw textual info about the device as produced by __tgt_get_device_info
    """
    _device_info_output[device_id] = info_text
    try:
        _device_info_map[device_id] = _parse_device_info(info_text)
    except Exception as e:
        raise RuntimeError(
            f"Warning: Failed to parse device info for device {device_id}: {e}"
        )


def _parse_device_info(output: str):
    """
    Parse the raw device info text to extract structured information about the
    device type, vendor, and architecture.

    :param output: Raw device info text as produced by __tgt_get_device_info
    :type output: str
    """
    pattern = re.compile(r"^(?P<key>[\w \-/\(\)]+?)\s{2,}(?P<val>.+)$")
    device_info_all = {}
    vendor = None
    devtype = None
    arch = None

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            continue
        key = m.group("key").strip().lower()
        val = m.group("val").strip().lower()

        device_info_all[key] = val

    if "amd" in device_info_all.get("vendor name", ""):
        vendor = "amd"
        devtype = "gpu"
        arch = device_info_all.get("device name", "")
    elif "nvidia" in device_info_all.get("device name", ""):
        vendor = "nvidia"
        devtype = "gpu"
        arch = device_info_all.get("compute capabilities", "")
    elif "generic-elf" in device_info_all.get("device type", ""):
        vendor = "host"
        devtype = "host"
        arch = "host"
    else:
        raise RuntimeError(
            f"Unable to determine device type/vendor from info: {device_info_all}"
        )

    return {"type": devtype, "vendor": vendor, "arch": arch}


def print_device_info(device_id: int):
    """
    Print the raw device info for a given device ID.

    :param device_id: Device ID as returned by OpenMP runtime
    :type device_id: int
    """
    info = _device_info_output.get(device_id, "No info available")
    print(f"Device {device_id} info:\n{info}")
    print("=" * 40)


def print_offloading_info():
    """
    Print the raw info about all OpenMP devices, and OpenMP device counts and
    defaults.
    """
    num_devices = omp_get_num_devices()
    for i in range(num_devices):
        print_device_info(i)

    print("omp_get_num_devices =", num_devices)
    print("omp_get_default_device =", omp_get_default_device())
    print("omp_get_initial_device =", omp_get_initial_device())


def find_device_ids(type=None, vendor=None, arch=None):
    """
    Return a list of device IDs matching the specified criteria. Any criteria
    that is None will be treated as a wildcard.

    :param type: Device type (e.g. "gpu", "host")
    :param vendor: Device vendor (e.g. "nvidia", "amd", "host")
    :param arch: Device architecture (e.g. "sm_80" for NVIDIA GPUs, or "gfx942" for AMD GPUs)
    """
    device_ids = []
    for k, v in _device_info_map.items():
        if type and v.get("type") != type:
            continue
        if vendor and v.get("vendor") != vendor:
            continue
        if arch and v.get("arch") != arch:
            continue
        device_ids.append(k)
    return device_ids


def get_device_type(device_id):
    """
    Get the device type for a given device ID.

    :param device_id: Device ID as returned by OpenMP runtime
    """
    info = _device_info_map.get(device_id)
    if info:
        return info.get("type")
    return None


def get_device_vendor(device_id):
    """
    Get the device vendor for a given device ID.

    :param device_id: Device ID as returned by OpenMP runtime
    """
    info = _device_info_map.get(device_id)
    if info:
        return info.get("vendor")
    return None


def get_device_arch(device_id):
    """
    Get the device architecture for a given device ID.

    :param device_id: Device ID as returned by OpenMP runtime
    """
    info = _device_info_map.get(device_id)
    if info:
        return info.get("arch")
    return None
