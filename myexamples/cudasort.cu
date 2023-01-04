#include <thrust/sort.h>
#include <thrust/device_vector.h>
// compile with nvcc -O2 -arch=sm_75 cudasort.cu -o cudasort
int main() {
  // Create a device vector to hold the data (error)
  //thrust::device_vector<int> data({3, 4, 2, 1, 5});
  std::vector<int> data2({3, 4, 2, 1, 5});
  thrust::device_vector<int> data(5,0);
  data=data2;
  
  // Sort the data in ascending order
  thrust::sort(data.begin(), data.end());

  
  // Print the sorted data
  for (int x : data) {
    std::cout << x << " ";
  }
  std::cout << std::endl;

  return 0;
}
