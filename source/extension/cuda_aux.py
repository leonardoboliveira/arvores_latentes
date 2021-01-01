import os

os.environ[
    "PATH"] += r";C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.24.28314\bin\Hostx64\x64;"
import pycuda.autoinit
import pycuda.driver as drv
import numpy
from datetime import datetime

from pycuda.compiler import SourceModule


class CudaAux:
    def __init__(self):
        mod = SourceModule("""
            __global__ void update_weights_nonnumeric(double *weights, double *cached_weights, int *nonnumeric_features, double update_val_for_weights, double update_val_for_cached_weights)
            {
              int col = blockIdx.x*blockDim.x+threadIdx.x;
              int row = blockIdx.y*blockDim.y+threadIdx.y;
              int index = col + row * 2000;

              weights[nonnumeric_features[index]] += update_val_for_weights;
              cached_weights[nonnumeric_features[index]] += update_val_for_cached_weights;
            }

            __global__ void update_weights_numeric(double *weights, double *cached_weights, int *numeric_features,float *numeric_vals, double update_val_for_weights, double update_val_for_cached_weights)
            {
              int col = blockIdx.x*blockDim.x+threadIdx.x;
              int row = blockIdx.y*blockDim.y+threadIdx.y;
              int index = col + row * 2000;

              weights[numeric_features[index]] += update_val_for_weights*numeric_vals[index];
              cached_weights[numeric_features[index]] += update_val_for_cached_weights*numeric_vals[index];        
            }
            """)

        self.update_weights_nonnumeric = mod.get_function("update_weights_nonnumeric")
        self.update_weights_numeric = mod.get_function("update_weights_numeric")

    def update_vector(self, weights,
                      cached_weights,
                      nonnumeric_features,
                      numeric_features,
                      numeric_vals,
                      update_val_for_weights,
                      update_val_for_cached_weights):
        total = nonnumeric_features.shape[0]
        if total > 0:
            x = max(total % 1024, 1)
            y = max(int(total / 1024), 1)
            self.update_weights_nonnumeric(drv.Out(weights), drv.Out(cached_weights), drv.In(nonnumeric_features),
                                           drv.In(update_val_for_weights), drv.In(update_val_for_cached_weights),
                                           block=(x, 1, 1), grid=(y, 1, 1))

        total = numeric_features.shape[0]
        if total > 0:
            x = max(total % 1024, 1)
            y = max(int(total / 1024), 1)
            self.update_weights_numeric(drv.Out(weights), drv.Out(cached_weights), drv.In(numeric_features),
                                        drv.In(numeric_vals), drv.In(update_val_for_weights),
                                        drv.In(update_val_for_cached_weights),
                                        block=(x, 1, 1), grid=(y, 1, 1))
