/********************************************************************************
 * The Host code running in the Arm CPU. Make sure to study and understand the
 *OpenCL API code
 *
 ********************************************************************************/
#include <assert.h>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iomanip> // For std::fixed and std::setprecision
#include <iostream>
#include <math.h>
#include <mutex>
#include <optional>
#include <queue>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>

static long long totalkernel_time = 0;
// XRT headers
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
// #include <experimental/xrt_queue.h>

#define TIC(timerName)                                                         \
  auto timerName##_start = std::chrono::high_resolution_clock::now()
#define TOC(timerName, elapsedVar)                                             \
  do {                                                                         \
    auto timerName##_end = std::chrono::high_resolution_clock::now();          \
    elapsedVar = std::chrono::duration_cast<std::chrono::microseconds>(        \
                     timerName##_end - timerName##_start)                      \
                     .count();                                                 \
  } while (0)

#define Tile_N 64    // Query size in a tile
#define Tile_M 65536 // Database size in a tile

const short GAP_i = -1;
const short GAP_d = -1;
const short MATCH = 2;
const short MISS_MATCH = -1;
const short CENTER = 0;
const short NORTH = 1;
const short NORTH_WEST = 2;
const short WEST = 3;

/**************
 *Multithread helper functions
 */
template <typename T> class ThreadSafeQueue {
public:
  void push(const T &item) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(item);
    cond_.notify_one();
  }

  // Blocks until an item is available, then returns it.
  std::optional<T> pop(const std::chrono::milliseconds &timeout_duration) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (cond_.wait_for(lock, timeout_duration,
                       [this]() { return !queue_.empty(); })) {
      T item = queue_.front();
      queue_.pop();
      return item; // Return the item if it was successfully popped
    } else {
      return std::nullopt; // Return an empty optional if timeout occurs
    }
  }

private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cond_{};
};

/***************************************************************************************
 * This is the golden code which runs in the CPU (and is the same code that you
 *developed for x86 / Arm) It will be used to verify the correct functionality
 *of the HW implementation. Its usefulness is mainly when you perform software
 *emulation (sw_emu).
 ***************************************************************************************/
void compute_matrices_sw(char *string1, char *string2, int *max_index,
                         int *similarity_matrix, short *direction_matrix,
                         int length_x, int length_y) {

  int index = 0;
  int i = 0;
  int j = 0;

  // Following values are used for the N, W, and NW values wrt.
  // similarity_matrix[i]
  int north = 0;
  int west = 0;
  int northwest = 0;

  int match = 0;
  int test_val = 0;
  int val = 0;
  int dir = 0;

  max_index[0] = 0;

  // Here the real computation starts. Place your code whenever is required.

  match = (string1[0] == string2[0]) ? MATCH : MISS_MATCH;

  if (match > 0) {
    similarity_matrix[0] = match;
    direction_matrix[0] = NORTH_WEST;
  }

  // First row

  for (index = 1; index < length_x; index++) {
    val = 0;
    west = similarity_matrix[index - 1] - 1;
    match = (string1[index] == string2[0]) ? MATCH : MISS_MATCH;
    dir = CENTER;
    if (west > 0) {
      val = west;
      dir = WEST;
    }
    if (match > west) {
      val = match;
      dir = NORTH_WEST;
    }
    similarity_matrix[index] = val;
    direction_matrix[index] = dir;
  }

  // Scan the N*M array row-wise starting from the second row.
  for (index = length_x; index < length_x * length_y; index++) {
    i = index % length_x; // column index
    j = index / length_x; // row index
    val = 0;
    dir = CENTER;

    if (i == 0) {
      // first column. (Checking only west and north)
      north = similarity_matrix[index - length_x] - 1;
      match = (string1[i] == string2[j]) ? MATCH : MISS_MATCH;
      dir = CENTER;
      if (north > 0) {
        val = north;
        dir = NORTH;
      }
      if (match > north) {
        val = match;
        dir = NORTH_WEST;
      }
    } else {
      // All other elements must compute all values (North, West and North-West)
      north = similarity_matrix[index - length_x] - 1;
      west = similarity_matrix[index - 1] - 1;
      northwest = similarity_matrix[index - length_x - 1];
      match = (string1[i] == string2[j]) ? MATCH : MISS_MATCH;
      test_val = northwest + match;
      val = 0;
      dir = CENTER;

      if (test_val > 0) {
        val = test_val;
        dir = NORTH_WEST;
      }

      if (north > val) {
        val = north;
        dir = NORTH;
      }

      if (west > val) {
        val = west;
        dir = WEST;
      }
    }

    if (val > similarity_matrix[max_index[0]]) {
      max_index[0] = index;
    }
    similarity_matrix[index] = val;
    direction_matrix[index] = dir;
  } // end of for-loop
} // end of function

void diagonal_to_simple(short *input_matrix, short *output_matrix) {

  int diagonal = 0;
  int elements_in_diag = 1;
  int out_index;
  int temp_row = 0;
  int temp_col = 0;
  int i = 0, j = 0;

  for (i = 0; i < ((Tile_N) * (Tile_N + Tile_M));) {
    if (diagonal < Tile_M) {
      temp_col = diagonal;
      temp_row = 0;
    } else {
      temp_row = (diagonal + 1) - Tile_M;
      temp_col = Tile_M - 1;
    }

    for (j = 0; j < elements_in_diag; j++) {
      out_index = temp_col * (Tile_N) + temp_row;
      output_matrix[out_index] = input_matrix[i];
      temp_col--;
      temp_row++;
      i++;
    }
    diagonal++;
    if (diagonal >= Tile_M) {
      elements_in_diag--;
      i = diagonal * Tile_N + Tile_N - elements_in_diag;
    } else {
      if (elements_in_diag != Tile_N) {
        elements_in_diag++;
        i = diagonal * Tile_N;
      }
    }
  }
}

int load_file_to_memory(const char *filename, char **result) {

  size_t size = 0;
  FILE *f = fopen(filename, "rb");
  if (f == NULL) {
    *result = NULL;
    return -1; // -1 means file opening fail
  }
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  fseek(f, 0, SEEK_SET);
  *result = (char *)malloc(size + 1);
  if (size != fread(*result, sizeof(char), size, f)) {
    free(*result);
    return -2; // -2 means file reading fail
  }
  fclose(f);
  (*result)[size] = 0;
  return size;
}

void fillFromFile(const char *filename, char *array, int size) {
  char *fileContent;
  int fileSize = load_file_to_memory(filename, &fileContent);
  if (fileSize < 0) {
    printf("Error: Failed to load file %s\n", filename);
    exit(EXIT_FAILURE);
  }
  strncpy(array, fileContent, size);
  free(fileContent);
}

/*************************************************************************
 * This structure is used to store the tile data.
 * It contains the last row, last column, last element,
 * and direction matrix for each tile.
 *************************************************************************/

struct TileComputationData {
  int i;
  int Ntiles;
  int Mtiles;
  int NumofTiles;
  int database_counter;
  std::vector<xrt::bo> &output_data;
  std::vector<xrt::bo> &input_data;
  std::vector<xrt::bo> &string_data;
  std::vector<xrt::bo> &direction_matrix;
  std::vector<char *> host_input_query;
  std::vector<char *> host_input_db;
  std::vector<short *> host_output_last_row;
  std::vector<short *> host_output_last_column;
  std::vector<short *> host_direction_matrix;
  std::vector<short *> host_input_prev_row;
  std::vector<short *> host_input_prev_column;
  std::vector<short *> host_input_prev_elem;
  std::vector<short *> host_output_last_elem;
  int CurrentCU;

  TileComputationData(int i_, int Ntiles_, int Mtiles_, int NumofTiles_,
                      int database_counter_, std::vector<xrt::bo> &output_data_,
                      std::vector<xrt::bo> &input_data_,
                      std::vector<xrt::bo> &string_data_,
                      std::vector<xrt::bo> &direction_matrix_,
                      std::vector<char *> host_input_query_,
                      std::vector<char *> host_input_db_,
                      std::vector<short *> host_output_last_row_,
                      std::vector<short *> host_output_last_column_,
                      std::vector<short *> host_direction_matrix_,
                      std::vector<short *> host_input_prev_row_,
                      std::vector<short *> host_input_prev_column_,
                      std::vector<short *> host_input_prev_elem_,
                      std::vector<short *> host_output_last_elem_,
                      int CurrentCU_)

      : i(i_), Ntiles(Ntiles_), Mtiles(Mtiles_), NumofTiles(NumofTiles_),
        database_counter(database_counter_), output_data(output_data_),
        input_data(input_data_), string_data(string_data_),
        direction_matrix(direction_matrix_),
        host_input_query(host_input_query_), host_input_db(host_input_db_),
        host_output_last_row(host_output_last_row_),
        host_output_last_column(host_output_last_column_),
        host_direction_matrix(host_direction_matrix_),
        host_input_prev_row(host_input_prev_row_),
        host_input_prev_column(host_input_prev_column_),
        host_input_prev_elem(host_input_prev_elem_),
        host_output_last_elem(host_output_last_elem_), CurrentCU(CurrentCU_) {}
};

struct Tiledata {
  TileComputationData data;
  int tile_id = -1;

  Tiledata(int tile_N, int tile_M, const TileComputationData &d) : data(d) {}
};

/*******************************************************************************
 *   This function executes a tile of the kernel.
 *   It sets the kernel arguments and runs the kernel for the specified tile.
 *******************************************************************************/
void execute_tile(const Tiledata &tile, xrt::kernel &krnl) {
  int cu = tile.data.CurrentCU;

  xrt::run run = xrt::run(krnl);
  run.set_arg(0, tile.data.string_data[cu]);
  run.set_arg(1, tile.data.input_data[cu]);
  run.set_arg(2, tile.data.output_data[cu]);
  run.set_arg(3, tile.data.direction_matrix[cu]);

  // Run the kernel
  TIC(runTimer);
  run.start();
  run.wait();
  long long runTime;
  TOC(runTimer, runTime);
  totalkernel_time += runTime;
}

void set_the_device_inputs(const Tiledata &tile, xrt::kernel &krnl, int i,
                           int j, int row, int col, int elem) {

  int cu = tile.data.CurrentCU;
  char *string_data_ptr = static_cast<char *>(tile.data.string_data[cu].map());
  if (!string_data_ptr) {
    printf("ERROR: Failed to map string_data[%d]\n", cu);
    exit(1);
  }
  memcpy(string_data_ptr, tile.data.host_input_query[j], Tile_N * sizeof(char));

  memcpy(string_data_ptr + Tile_N, tile.data.host_input_db[i],
         (Tile_M + 2 * (Tile_N - 1)) * sizeof(char));

  tile.data.string_data[cu].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  short *input_data_ptr = static_cast<short *>(tile.data.input_data[cu].map());

  // Copy prev_row_matrix
  memcpy(input_data_ptr, tile.data.host_output_last_row[row],
         Tile_N * (Tile_M + Tile_N - 1) * sizeof(short));

  // Copy prev_column_matrix
  memcpy(input_data_ptr + Tile_N * (Tile_M + Tile_N - 1),
         tile.data.host_output_last_column[col],
         (Tile_M + Tile_N - 1) * sizeof(short));

  // Copy prev_elem
  *(input_data_ptr + Tile_N * (Tile_M + Tile_N - 1) + (Tile_M + Tile_N - 1)) =
      *(tile.data.host_output_last_elem[elem]);
  tile.data.input_data[cu].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  execute_tile(tile, krnl);
}

void propagate_tile_boundaries(const Tiledata &tile) {
  int i = tile.data.i;
  int Ntiles = tile.data.Ntiles;
  int NumofTiles = tile.data.NumofTiles;
  int CurrentCU = tile.data.CurrentCU;

  // Sync and copy direction matrix
  tile.data.direction_matrix[CurrentCU].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  short *mapped_ptr =
      static_cast<short *>(tile.data.direction_matrix[CurrentCU].map());
  memcpy(tile.data.host_direction_matrix[i], mapped_ptr,
         sizeof(short) * (Tile_N * (Tile_N + Tile_M)));

  tile.data.output_data[CurrentCU].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  short *output_ptr =
      static_cast<short *>(tile.data.output_data[CurrentCU].map());
  if (!output_ptr) {
    printf("ERROR: Failed to map output_data[%d]\n", CurrentCU);
    exit(1);
  }

  // Offsets in unified output buffer
  size_t last_row_offset = 0;
  size_t last_col_offset = Tile_M + Tile_N - 1;
  size_t last_elem_offset = 2 * (Tile_M + Tile_N - 1);

  // Prepare next column if needed
  if ((i % Ntiles != Ntiles - 1) && (i + 1 <= NumofTiles)) {
    std::copy(output_ptr + last_col_offset + Tile_N - 1,
              output_ptr + last_col_offset + (Tile_M + Tile_N - 1),
              tile.data.host_output_last_column[i]);
    std::fill(tile.data.host_output_last_column[i] + (Tile_N - 1) + Tile_M,
              tile.data.host_output_last_column[i] + (Tile_M + Tile_N - 1), 0);
  }
  // Prepare next row if needed
  if (i + Ntiles < NumofTiles) {
    std::fill(tile.data.host_output_last_row[i],
              tile.data.host_output_last_row[i] + Tile_N * (Tile_N + 1), 0);
    for (int k = 0; k < Tile_N; ++k) {
      tile.data.host_output_last_row[i][k * (Tile_N + 1)] =
          output_ptr[last_row_offset + (Tile_M - 1) + k];
    }
  }
  // Propagate prev element diagonally if needed
  if ((i % Ntiles != Ntiles - 1) && (i + Ntiles + 1 <= NumofTiles)) {
    *tile.data.host_output_last_elem[i] = output_ptr[last_elem_offset];
  }
}

/********************** Worker thread function ************************/
//------------------------------------------------------------------------------
// Function executed by each CU thread.
// Each thread uses its own thread-safe job queue and handles its own cu.
void cu_thread_func(ThreadSafeQueue<Tiledata *> &job_queue, xrt::kernel &krnl,
                    int i, int j, int Ntiles, int NumofTiles) {
  auto tiledata_opt = job_queue.pop(std::chrono::milliseconds(1000000));

  if (tiledata_opt) {
    Tiledata *tiledata = *tiledata_opt;

    try {
      int up_i = i - 1, up_j = j;         // Top tile
      int left_i = i, left_j = j - 1;     // Left tile
      int diag_i = i - 1, diag_j = j - 1; // Top-left tile

      bool has_up = up_i >= 0;
      bool has_left = left_j >= 0;
      bool has_diag = diag_i >= 0 && diag_j >= 0;

      int row_idx = has_up ? (up_i * Ntiles + up_j) : NumofTiles - 1;
      int col_idx = has_left ? (left_i * Ntiles + left_j) : NumofTiles - 1;
      int elem_idx = has_diag ? (diag_i * Ntiles + diag_j) : NumofTiles - 1;

      set_the_device_inputs(*tiledata, krnl, i, j, row_idx, col_idx,
                            elem_idx); // Set the inputs for the current tile
      propagate_tile_boundaries(*tiledata); // Propagate boundaries if needed
    } catch (const std::exception &e) {
      std::cerr << "Exception in CU thread: " << e.what();
    }
  }
}

/*******************************************************************************
 *   Host program running on the Arm CPU.
 *   The code is written using the OpenCL API.
 *   We have provided multiple comments for you to understand where each thing
 *******************************************************************************/
int main(int argc, char **argv) {
  printf("\nstarting HOST code \n");

  if (argc < 6) {
    printf("%s <input xclbin file>\n", argv[0]);
    return EXIT_FAILURE;
  }
  int N = atoi(argv[4]);
  int M = atoi(argv[5]);

  if (N <= 0 || M <= 0) {
    printf("N and M should be positive numbers. \n");
    return EXIT_FAILURE;
  }

  int Ntiles = (N / Tile_N);
  int Mtiles = (M / Tile_M);
  int NumofCUs = 8;
  int NumofTiles = Ntiles * Mtiles;
  printf("NumofTiles = %d\n", NumofTiles);
  printf("\n\nN = %d , M = %d , Tile_N = %d , Tile_M = %d,"
         "Ntiles = %d  and Mtiles = %d\n\n",
         N, M, Tile_N, Tile_M, Ntiles, Mtiles);
  fflush(stdout);

  /*host side buffers*/

  char *query = (char *)malloc(sizeof(char) * N);
  char *database = (char *)malloc(sizeof(char) * M);
  if (!query || !database) {
    fprintf(stderr, "Memory allocation failed for query or database\n");
    return EXIT_FAILURE;
  }
  std::vector<char *> host_input_db(Mtiles);
  for (int i = 0; i < Mtiles; ++i) {
    host_input_db[i] = new char[(Tile_M + 2 * (Tile_N - 1))]();
  }

  std::vector<char *> host_input_query(Ntiles);
  for (int i = 0; i < Ntiles; ++i)
    host_input_query[i] = new char[Tile_N * sizeof(char)]();

  std::vector<short *> host_input_prev_row(NumofTiles);
  for (int i = 0; i < NumofTiles; ++i)
    host_input_prev_row[i] = new short[Tile_N * (Tile_M + Tile_N - 1)]();

  std::vector<short *> host_input_prev_column(NumofTiles);
  for (int i = 0; i < NumofTiles; ++i)
    host_input_prev_column[i] = new short[Tile_M + Tile_N - 1]();

  std::vector<short *> host_input_prev_elem(NumofTiles);
  for (int i = 0; i < NumofTiles; ++i)
    host_input_prev_elem[i] = new short(0);

  std::vector<short *> host_direction_matrix(NumofTiles);
  for (int i = 0; i < NumofTiles; ++i)
    host_direction_matrix[i] = new short[Tile_N * (Tile_N + Tile_M)]();

  std::vector<short *> host_output_last_row(NumofTiles);
  for (int i = 0; i < NumofTiles; ++i)
    host_output_last_row[i] = new short[Tile_N * (Tile_N + Tile_M - 1)]();

  std::vector<short *> host_output_last_column(NumofTiles);
  for (int i = 0; i < NumofTiles; ++i)
    host_output_last_column[i] = new short[Tile_M + Tile_N - 1]();

  std::vector<short *> host_output_last_elem(NumofTiles);
  for (int i = 0; i < NumofTiles; ++i)
    host_output_last_elem[i] = new short(0);

  auto device = xrt::device(0);
  auto uuid = device.load_xclbin(argv[1]);
  std::vector<xrt::kernel> krnl;
  for (int i = 1; i <= NumofCUs; ++i) {
    krnl.emplace_back(device, uuid,
                      "compute_matrices:{compute_matrices" + std::to_string(i) +
                          "}");
  }
  printf("Available kernels:\n");
  for (size_t i = 0; i < krnl.size(); ++i) {
    std::cout << "  Kernel " << i << ": " << krnl[i].get_name() << std::endl;
  }

  size_t sz_query = Tile_N * sizeof(char);
  size_t sz_db = (Tile_M + 2 * (Tile_N - 1)) * sizeof(char);
  size_t sz_row = Tile_N * (Tile_M + Tile_N - 1) * sizeof(short);
  size_t sz_col = (Tile_M + Tile_N - 1) * sizeof(short);
  size_t sz_elem = sizeof(short);
  size_t sz_dir = Tile_N * (Tile_N + Tile_M) * sizeof(short);

  /*create the devise-visible buffers*/

  std::vector<xrt::bo> direction_matrix(NumofCUs);
  std::vector<xrt::bo> string_data(NumofCUs);
  std::vector<xrt::bo> input_data(NumofCUs);
  std::vector<xrt::bo> output_data(NumofCUs);

  std::vector<Tiledata> tile_outputs;
  tile_outputs.reserve(NumofTiles);
  for (int idx = 0; idx < NumofTiles; ++idx) {
    TileComputationData d(
        idx, Ntiles, Mtiles, NumofTiles, idx % Mtiles, output_data, input_data,
        string_data, direction_matrix, host_input_query, host_input_db,
        host_output_last_row, host_output_last_column, host_direction_matrix,
        host_input_prev_row, host_input_prev_column, host_input_prev_elem,
        host_output_last_elem, idx % NumofCUs);
    // Create a Tiledata object for each tile
    tile_outputs.emplace_back(Tile_N, Tile_M, d);
  }

  // Create one thread-safe queue per CU thread.
  std::vector<ThreadSafeQueue<Tiledata *>> queues(NumofCUs);
  std::vector<ThreadSafeQueue<Tiledata *> *> queue_ptrs;
  for (auto &q : queues) {
    queue_ptrs.push_back(&q);
  }

  for (int i = 0; i < NumofCUs; i++) {
    string_data[i] = xrt::bo(device, sz_query + sz_db, krnl[i].group_id(0));
    input_data[i] =
        xrt::bo(device, sz_row + sz_col + sz_elem, krnl[i].group_id(1));
    output_data[i] = xrt::bo(device, 2 * sz_col + sz_elem, krnl[i].group_id(2));
    direction_matrix[i] = xrt::bo(device, sz_dir, krnl[i].group_id(3));
  }
  fillFromFile(argv[2], query, N);
  fillFromFile(argv[3], database, M);

  printf("fillFromFile done!\n");
  fflush(stdout);

  for (int i = 0; i < Mtiles; i++) {
    // Step 1: Add 3 'I's at the beginning of each split
    for (int j = 0; j < (Tile_N - 1); j++) {
      host_input_db[i][j] = 'I';
    }

    // Step 2: Copy the relevant data from the original database into the
    // split
    memcpy(&host_input_db[i][(Tile_N - 1)], &database[i * Tile_M], Tile_M);

    // Step 3: Add 3 'I's at the end of each split
    for (int j = Tile_N - 1 + Tile_M; j < (Tile_M + 2 * (Tile_N - 1)); j++) {
      host_input_db[i][j] = 'I';
    }
  }

  for (int i = 0; i < Ntiles; i++) {
    memcpy(host_input_query[i], query + i * Tile_N, Tile_N * sizeof(char));
  }
  printf("Input data prepared!\n");

  TIC(hwTimer);
  std::vector<std::thread> cu_threads;
  int cu = 0;
  for (int wave = 0; wave < Mtiles + Ntiles - 1; wave++) {
    fflush(stdout);

    std::vector<std::thread> cu_threads;
    for (int i = 0; i <= wave && i < Mtiles; i++) {
      int j = wave - i;
      if (j >= Ntiles)
        continue;
      int tileIndex = i * Ntiles + j;
      if (tileIndex >= NumofTiles) {
        continue;
      }
      if (cu == NumofCUs) {
        for (auto &t : cu_threads) {
          if (t.joinable())
            t.join();
        }
        cu = 0;
      }
      tile_outputs[tileIndex].data.CurrentCU = cu;
      tile_outputs[tileIndex].tile_id = tileIndex;
      queues[cu].push(&tile_outputs[tileIndex]);
      cu_threads.emplace_back(cu_thread_func, std::ref(queues[cu]),
                              std::ref(krnl[cu]), i, j, Ntiles, NumofTiles);
      cu++;
    }
    for (auto &t : cu_threads) {
      if (t.joinable())
        t.join();
    }
  }

  long long hwTime;
  TOC(hwTimer, hwTime);
  std::cout << "Total hw time : " << hwTime << " us (" << std::fixed
            << std::setprecision(4) << hwTime / 1e6 << " seconds)" << std::endl;

#ifdef sw_validation
  // rearrange the direction matrices from the diagonal form to a linear one
  short *hw_data_ptrs[NumofTiles];
  for (int wave = 0; wave < Mtiles + Ntiles - 1; wave++) {
    for (int i = 0; i <= wave && i < Mtiles; i++) {
      int j = wave - i;
      if (j >= Ntiles)
        continue;
      int tileIndex = i * Ntiles + j;
      if (tileIndex >= NumofTiles) {
        printf("Tile index %d exceeds NumofTiles %d\n", tileIndex, NumofTiles);
        continue;
      }
      hw_data_ptrs[tileIndex] =
          tile_outputs[tileIndex].data.host_direction_matrix[tileIndex];
      if (!hw_data_ptrs[tileIndex]) {
        fprintf(stderr, "Error mapping tile %d\n", tileIndex);
        exit(EXIT_FAILURE);
      }
      diagonal_to_simple(hw_data_ptrs[tileIndex], hw_data_ptrs[tileIndex]);
    }
  }
#endif
#ifdef sw_validation
  /**************************************************************
   * Run the same algorithm in the Host Unit and compare for verification
   **************************************************************/

  printf("\nStarting SW computation \n");
  fflush(stdout);
  int *similarity_matrix_sw = (int *)malloc(sizeof(int) * (N) * (M));
  short *direction_matrix_sw = (short *)malloc(sizeof(short) * (N) * (M));
  int *max_index_sw = (int *)malloc(sizeof(int));

  for (int i = 0; i < (N) * (M); i++) {
    similarity_matrix_sw[i] = 0;
    direction_matrix_sw[i] = 0;
  }

  TIC(swTimer); // Start time

  compute_matrices_sw(query, database, max_index_sw, similarity_matrix_sw,
                      direction_matrix_sw, N, M);

  long long swTime;
  TOC(swTimer, swTime); // End time
  std::cout << "\nSW computation ended in " << swTime << " us (" << std::fixed
            << std::setprecision(4) << swTime / 1e6 << " seconds)" << std::endl;

  // Calculate HW speedup
  double hwSpeedup = static_cast<double>(swTime) / hwTime;
  std::cout << "HW Speedup: " << std::fixed << std::setprecision(2) << hwSpeedup
            << "x" << std::endl;

  /**************************************************************
   * Compare the results of the software and hardware
   **************************************************************/
  printf("comparing the results\n");
  for (int tileIndex = 0; tileIndex < NumofTiles; tileIndex++) {
    for (int i = 0; i < Tile_M; i++) {
      for (int j = 0; j < Tile_N; j++) {
        int globalRow = (tileIndex / Ntiles) * Tile_M + i;
        int globalCol = (tileIndex % Ntiles) * Tile_N + j;
        if (direction_matrix_sw[globalRow * N + globalCol] !=
            hw_data_ptrs[tileIndex][i * Tile_N + j]) {
          printf("Error, mismatch in the results at tile %d, row %d, col %d, "
                 "SW: %d, HW: %d\n",
                 tileIndex, globalRow, globalCol,
                 direction_matrix_sw[globalRow * N + globalCol],
                 hw_data_ptrs[tileIndex][i * Tile_N + j]);
          return EXIT_FAILURE;
        }
      }
    }
  }
  printf("Success: computation ended!- RESULTS CORRECT\n");
  free(similarity_matrix_sw);
  free(direction_matrix_sw);
  free(max_index_sw);
#endif
  free(query);
  free(database);
  for (int i = 0; i < Mtiles; ++i)
    delete[] host_input_db[i];
  for (int i = 0; i < Ntiles; ++i)
    delete[] host_input_query[i];
  for (int i = 0; i < NumofTiles; ++i) {
    delete[] host_input_prev_row[i];
    delete[] host_input_prev_column[i];
    delete[] host_direction_matrix[i];
    delete[] host_output_last_row[i];
    delete[] host_output_last_column[i];
    delete[] host_input_prev_elem[i];
    delete[] host_output_last_elem[i];
  }

  return EXIT_SUCCESS;
}
