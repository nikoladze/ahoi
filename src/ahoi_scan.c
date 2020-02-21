#include <stddef.h>

void fill_matching(char **masks, double wi, size_t j, size_t combination_index,
                   int index_factor, size_t *shape, size_t ndims, long *counts,
                   double *sumw, double *sumw2, char use_weights) {
  for (int i = 0; i < shape[j]; ++i) {
    if (i > 0) {
      combination_index += index_factor;
    }
    if (!masks[j][i]) {
      continue;
    }
    if (j != (ndims - 1)) {
      fill_matching(masks, wi, j + 1, combination_index,
                    index_factor / shape[j + 1], shape, ndims, counts, sumw,
                    sumw2, use_weights);
    } else {
      counts[combination_index] += 1;
      if (use_weights) {
        sumw[combination_index] += wi;
        sumw2[combination_index] += wi * wi;
      }
    }
  }
}
