#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void fillRandom(char *string, size_t dimension) {
  static const char possibleLetters[] = "ATCG";

  for (size_t i = 0; i < dimension; i++) {
    int randomNum = rand() % 4;
    string[i] = possibleLetters[randomNum];
  }
}

void createFileWithRandomContent(const char *filename, size_t size) {
  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    perror("Error opening file");
    return;
  }

  // Instead of allocating the whole thing (impossible for billions),
  // generate & write in chunks
  const size_t CHUNK = 1024 * 1024; // 1 MB
  char *buffer = malloc(CHUNK);
  if (buffer == NULL) {
    perror("Error allocating memory");
    fclose(file);
    return;
  }

  for (size_t written = 0; written < size;) {
    size_t toWrite = (size - written > CHUNK) ? CHUNK : (size - written);
    fillRandom(buffer, toWrite);
    fwrite(buffer, 1, toWrite, file);
    written += toWrite;
  }

  free(buffer);
  fclose(file);
}

int main(int argc, char *argv[]) {
  srand(time(NULL));
  long long querySize;
  long long databaseSize;

  if (argc != 3) {
    printf("Usage: %s <querySize> <databaseSize>\n", argv[0]);
    return 1;
  } else {
    querySize = atoll(argv[1]);
    databaseSize = atoll(argv[2]);
    printf("querySize: %lld\n", querySize);
    printf("databaseSize: %lld\n", databaseSize);
  }

  createFileWithRandomContent("query.txt", querySize);
  createFileWithRandomContent("database.txt", databaseSize);

  return 0;
}
