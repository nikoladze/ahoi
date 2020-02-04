#include <stddef.h>

int ravel_multi_index(int *multi_index, int *shape, int ndims) {
  int i_1d = multi_index[ndims - 1];
  int k = shape[ndims - 1];
  for (int i = ndims - 2; i >= 0; --i) {
    i_1d += multi_index[i] * k;
    k *= shape[i];
  }
  return i_1d;
}

void fill_matching(char **masks, double wi, int j, int *multi_index, int *shape,
                   size_t ndims, long *counts, double *sumw, double *sumw2) {
  int combination_index;
  for (int i = 0; i < shape[j]; ++i) {
    if (!masks[j][i]) {
      continue;
    }
    multi_index[j] = i;
    if (j != (ndims - 1)) {
      fill_matching(masks, wi, j + 1, multi_index, shape, ndims, counts, sumw, sumw2);
    } else {
      combination_index = ravel_multi_index(multi_index, shape, ndims);
      counts[combination_index] += 1;
      sumw[combination_index] += wi;
      sumw2[combination_index] += wi * wi;
    }
  }
}
