#include <stddef.h>

int ravel_multi_index(int *inds, int *dims, int ndims) {
  int i_1d = inds[ndims - 1];
  int k = dims[ndims - 1];
  for (int i = ndims - 2; i >= 0; --i) {
    i_1d += inds[i] * k;
    k *= dims[i];
  }
  return i_1d;
}

void check_fill(char **masks, double wi, int j, int *inds, int *dims,
                size_t ndims, long *counts, double *sumw, double *sumw2) {
  int combination_index;
  for (int i = 0; i < dims[j]; ++i) {
    if (!masks[j][i]) {
      continue;
    }
    inds[j] = i;
    if (j != (ndims - 1)) {
      check_fill(masks, wi, j + 1, inds, dims, ndims, counts, sumw, sumw2);
    } else {
      combination_index = ravel_multi_index(inds, dims, ndims);
      counts[combination_index] += 1;
      sumw[combination_index] += wi;
      sumw2[combination_index] += wi * wi;
    }
  }
}
