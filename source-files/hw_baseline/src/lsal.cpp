#include <ap_int.h>
#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

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

extern "C" {
void compute_matrices(
    char *string_data,       // string1 + string2
    short *input_data,       // prev_row_matrix + prev_column_matrix + prev_elem
    short *output_data,      // last_row_matrix + last_column_matrix + last_elem
    short *direction_matrix) // just direction matrix
{

#pragma HLS INTERFACE m_axi port = string_data offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = input_data offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = output_data offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = direction_matrix offset = slave bundle =    \
    gmem4

// Add control interfaces:
#pragma HLS INTERFACE s_axilite port = string_data
#pragma HLS INTERFACE s_axilite port = input_data
#pragma HLS INTERFACE s_axilite port = output_data
#pragma HLS INTERFACE s_axilite port = direction_matrix
#pragma HLS INTERFACE s_axilite port = return

  int p, t, i;
  int north = 0, west = 0, northwest = 0;
  int direction_index = 0;
  int match = 0, test_val = 0, val = 0, dir = 0;
  int max_value_temp = 0, max_index_temp = 0, diag_length = Tile_N;

  int max_value_matrix[Tile_N];
#pragma HLS ARRAY_PARTITION variable = max_value_matrix dim = 1 type = complete
  int max_index_matrix[Tile_N];
#pragma HLS ARRAY_PARTITION variable = max_index_matrix dim = 1 type = complete

  int *temp = NULL;
  int col_p = 0;
  int row_p = 0;
  int row = -Tile_M;
  short temp_val;

  char *string1 = string_data;
  char *string2 = &string_data[Tile_N];

  short *prev_row_matrix = input_data;
  short *prev_column_matrix = &input_data[Tile_N * (Tile_M + Tile_N - 1)];
  short *prev_elem =
      &input_data[Tile_N * (Tile_M + Tile_N - 1) + (Tile_M + Tile_N - 1)];

  short *last_row_matrix = output_data;
  short *last_column_matrix = &output_data[Tile_M + Tile_N - 1];
  short *last_elem = &output_data[2 * (Tile_M + Tile_N - 1)];

  int similarity_matrix_buffer1[Tile_N + 1];
#pragma HLS ARRAY_PARTITION dim = 1 type = complete variable =                 \
    similarity_matrix_buffer1
  int similarity_matrix_buffer2[Tile_N + 1];
#pragma HLS ARRAY_PARTITION dim = 1 type = complete variable =                 \
    similarity_matrix_buffer2
  int similarity_matrix_buffer3[Tile_N + 1];
#pragma HLS ARRAY_PARTITION dim = 1 type = complete variable =                 \
    similarity_matrix_buffer3

  short direction_matrix_buffer[2 * Tile_N];
#pragma HLS ARRAY_PARTITION variable = direction_matrix_buffer dim =           \
    1 factor = 8 cyclic

  memset(similarity_matrix_buffer1, 0, sizeof(int) * (Tile_N + 1));
  memset(similarity_matrix_buffer2, 0, sizeof(int) * (Tile_N + 1));
  memset(similarity_matrix_buffer3, 0, sizeof(int) * (Tile_N + 1));
  memset(direction_matrix_buffer, 0, sizeof(short) * (2 * Tile_N));

  memset(max_value_matrix, 0, sizeof(int) * (Tile_N));
  memset(max_index_matrix, 0, sizeof(int) * (Tile_N));
  memset(last_column_matrix, 0, sizeof(short) * (Tile_M + Tile_N - 1));
  memset(last_row_matrix, 0, sizeof(short) * (Tile_M + Tile_N - 1));

  char string1_buffer[Tile_N];
  char string2_buffer[Tile_N + 1];
#pragma HLS ARRAY_PARTITION dim = 1 type = complete variable = string2_buffer

  memcpy(string1_buffer, string1, (Tile_N) * sizeof(char));
  memcpy(string2_buffer, string2, (Tile_N + 1) * sizeof(char));

  similarity_matrix_buffer1[0] = *prev_elem;
Diag_Num_Loop:
  for (p = Tile_N; p < (Tile_M + 2 * (Tile_N - 1) + 1); p++) {
#pragma HLS PIPELINE
    similarity_matrix_buffer2[0] = prev_column_matrix[col_p];
    col_p++;
  Diag_Length_Loop:
    for (t = 0; t < Tile_N; t++) {
#pragma HLS UNROLL
      direction_index = p - Tile_N;

      similarity_matrix_buffer2[t + 1] =
          (similarity_matrix_buffer2[t + 1] & ~-(!!prev_row_matrix[row_p])) |
          (prev_row_matrix[row_p] & -!!prev_row_matrix[row_p]);

      row_p++;

      north = similarity_matrix_buffer2[t + 1] - 1;
      west = similarity_matrix_buffer2[t] - 1;
      northwest = similarity_matrix_buffer1[t];

      bool condition = string1_buffer[t] == string2_buffer[Tile_N - t - 1];
      match = (condition * MATCH) + (!condition * MISS_MATCH);
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

      if (val > max_value_matrix[t]) {
        max_index_matrix[t] = (direction_index - t) * Tile_N + i;
        max_value_matrix[t] = val;
      }
      similarity_matrix_buffer3[t + 1] = val;
      direction_matrix_buffer[(p % 2) * Tile_N + t] = dir;
    } // end of inner for-loop

    last_row_matrix[Tile_M + row] = similarity_matrix_buffer3[row + 2];
    last_column_matrix[row + Tile_M] = similarity_matrix_buffer3[Tile_N];
    row++;

    memmove(string2_buffer, string2_buffer + 1, (Tile_N - 1) * sizeof(char));
    string2_buffer[Tile_N - 1] = string2[p];

    memcpy(similarity_matrix_buffer1, similarity_matrix_buffer2,
           sizeof(int) * (Tile_N + 1));
    memcpy(similarity_matrix_buffer2, similarity_matrix_buffer3,
           sizeof(int) * (Tile_N + 1));
    memcpy(&direction_matrix[direction_index * Tile_N],
           direction_matrix_buffer + (p % 2) * Tile_N,
           sizeof(short) * (Tile_N));
  } // end of outer for-loop

  *last_elem = last_row_matrix[Tile_M + Tile_N - 2];

  // Calculate max value and index
  for (i = 0; i < diag_length; i++) {
    if (max_value_matrix[i] > max_value_temp) {
      max_value_temp = max_value_matrix[i];
      max_index_temp = max_index_matrix[i];
    }
  }
}
}
