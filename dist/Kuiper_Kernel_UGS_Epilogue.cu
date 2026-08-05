
#include "Kuiper_Kernel_UGS_Epilogue.h"

krml_checked_int_t Kuiper_Kernel_UGS_Epilogue_pair_group_n = 64;

krml_checked_int_t Kuiper_Kernel_UGS_Epilogue_gate_col_of(krml_checked_int_t c)
{
    return
        Prims_op_Addition(Prims_op_Star(128, Prims_op_Division(c, 64)),
                          Prims_op_Modulus(c, 64));
}

krml_checked_int_t Kuiper_Kernel_UGS_Epilogue_up_col_of(krml_checked_int_t c)
{
    return
        Prims_op_Addition(Prims_op_Addition
                          (Prims_op_Star(128, Prims_op_Division(c, 64)),
                           Prims_op_Modulus(c, 64)), 64);
}

krml_checked_int_t
Kuiper_Kernel_UGS_Epilogue_gate_col_idx(krml_checked_int_t n,
                                        krml_checked_int_t c)
{
    KRML_MAYBE_UNUSED_VAR(n);
    return
        Prims_op_Addition(Prims_op_Star(128, Prims_op_Division(c, 64)),
                          Prims_op_Modulus(c, 64));
}

krml_checked_int_t
Kuiper_Kernel_UGS_Epilogue_up_col_idx(krml_checked_int_t n,
                                      krml_checked_int_t c)
{
    KRML_MAYBE_UNUSED_VAR(n);
    return
        Prims_op_Addition(Prims_op_Addition
                          (Prims_op_Star(128, Prims_op_Division(c, 64)),
                           Prims_op_Modulus(c, 64)), 64);
}

bool
Kuiper_Kernel_UGS_Epilogue_converted_before(krml_checked_int_t n,
                                            krml_checked_int_t k,
                                            krml_checked_int_t r,
                                            krml_checked_int_t c)
{
    return Prims_op_LessThan(Prims_op_Addition(Prims_op_Star(r, n), c), k);
}
