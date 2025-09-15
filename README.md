# Prototype of Intel Plugin for TorchCodec

This repos contains a prototype of the Intel Plugin for [TorchCodec]. It
requires https://github.com/meta-pytorch/torchcodec/pull/938 to work.

Import the project in your Python script to register Intel device
interface in the TorchCodec:

```
import torchcodec
import torchcodec_xpu
```

[TorchCodec]: https://github.com/meta-pytorch/torchcodec
