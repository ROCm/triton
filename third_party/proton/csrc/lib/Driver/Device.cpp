#include "Driver/Device.h"
#include "Driver/GPU/HipApi.h"

#include "Utility/Errors.h"

namespace proton {

Device getDevice(DeviceType type, uint64_t index) {
  if (type == DeviceType::HIP) {
    return hip::getDevice(index);
  }
  throw std::runtime_error("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
  if (type == DeviceType::HIP) {
    return DeviceTraits<DeviceType::HIP>::name;
  }
  throw std::runtime_error("DeviceType not supported");
}

} // namespace proton
