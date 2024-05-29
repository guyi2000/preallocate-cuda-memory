# Preallocate CUDA memory for pytorch

This is a code that helps preallocate memory for PyTorch, used for competing for computational resources with others.

You can use the following command directly on the command line:

```bash
python -m preallocate_cuda_memory
```

Or you can use in python file:

```python
import preallocate_cuda_memory as pc

mc = pc.MemoryController(0)  # 0 is the GPU index
mc.occupy_all_available_memory()
mc.free_memory()
```

If you find any issues, please feel free to contact the author by raising an issue on GitHub.
