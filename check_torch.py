import torch

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'PyTorch version: {torch.__version__}')
print(f'Device count: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('Device name: None')
    print('CUDA version: None')
