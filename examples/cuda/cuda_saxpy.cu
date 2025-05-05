// This program performs a simple single-precision Ax+Y operation
// using cudaFlow and verifies its result.

#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>

// Kernel: saxpy
__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a*x[i] + y[i];
  }
}

// Function: main
int main() {
  
  const unsigned N = 1<<20;

  std::vector<float> hx, hy;

  float* dx {nullptr};
  float* dy {nullptr};
  
  // allocate x
  hx.resize(N, 1.0f);
  cudaMalloc(&dx, N*sizeof(float));

  // allocate y
  hy.resize(N, 2.0f);
  cudaMalloc(&dy, N*sizeof(float));
  
  // saxpy cudaFlow
  tf::cudaFlow cf;
  auto h2d_x = cf.copy(dx, hx.data(), N);
  auto h2d_y = cf.copy(dy, hy.data(), N);
  auto d2h_x = cf.copy(hx.data(), dx, N);
  auto d2h_y = cf.copy(hy.data(), dy, N);
  auto kernel = cf.kernel((N+255)/256, 256, 0, saxpy, N, 2.0f, dx, dy);
  kernel.succeed(h2d_x, h2d_y)
        .precede(d2h_x, d2h_y);
  
  tf::cudaStream stream;
  tf::cudaGraphExec exec(cf);
  exec.run(stream);
  stream.synchronize();
  
  // visualize this cudaflow
  cf.dump(std::cout);

  // Add a verification task
  float max_error = 0.0f;
  for (size_t i = 0; i < N; i++) {
    max_error = std::max(max_error, abs(hx[i]-1.0f));
    max_error = std::max(max_error, abs(hy[i]-4.0f));
  }
  std::cout << "saxpy finished with max error: " << max_error << '\n';

  // free memory
  cudaFree(dx);
  cudaFree(dy);

  return 0;
}

