
#ifndef Kuiper_Kernel_UGS_Epilogue_H
#define Kuiper_Kernel_UGS_Epilogue_H

#include <kuiper.h>

extern krml_checked_int_t Kuiper_Kernel_UGS_Epilogue_pair_group_n;

krml_checked_int_t Kuiper_Kernel_UGS_Epilogue_gate_col_of(krml_checked_int_t c);

krml_checked_int_t Kuiper_Kernel_UGS_Epilogue_up_col_of(krml_checked_int_t c);

krml_checked_int_t
Kuiper_Kernel_UGS_Epilogue_gate_col_idx(krml_checked_int_t n,
                                        krml_checked_int_t c);

krml_checked_int_t
Kuiper_Kernel_UGS_Epilogue_up_col_idx(krml_checked_int_t n,
                                      krml_checked_int_t c);

bool
Kuiper_Kernel_UGS_Epilogue_converted_before(krml_checked_int_t n,
                                            krml_checked_int_t k,
                                            krml_checked_int_t r,
                                            krml_checked_int_t c);

#define Kuiper_Kernel_UGS_Epilogue_H_DEFINED
#endif                          /* Kuiper_Kernel_UGS_Epilogue_H */
