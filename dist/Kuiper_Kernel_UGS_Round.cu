
#include "Kuiper_Kernel_UGS_Round.h"

bool
Kuiper_Kernel_UGS_Round_converted_before(krml_checked_int_t n,
                                         krml_checked_int_t k,
                                         krml_checked_int_t r,
                                         krml_checked_int_t c)
{
    return Prims_op_LessThan(Prims_op_Addition(Prims_op_Star(r, n), c), k);
}
