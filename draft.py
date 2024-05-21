import onnxruntime as ort

# Check the available execution providers
available_providers = ort.get_available_providers()
print("Available execution providers:", available_providers)