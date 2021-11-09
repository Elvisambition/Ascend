import math
import numpy
from tbe import tik


def softmax_tik(input_1, input_2):


    tik_instance = tik.Tik()

    shape = input_1.get("ori_shape")
    data_type = input_2.get("dtype").lower()


    length = shape[0]
    W = shape[1]


    x_gm = tik_instance.Tensor(data_type, shape, scope=tik.scope_gm, name="x_gm")
    y_gm = tik_instance.Tensor(data_type, shape, scope=tik.scope_gm, name="w_gm")
    output_gm = tik_instance.Tensor(data_type, shape, scope=tik.scope_gm, name="result_gm")


    with tik_instance.for_range(0, length) as i:

        x_ub = tik_instance.Tensor(data_type, (length, W), scope=tik.scope_ubuf, name="x_ub")
        w_ub = tik_instance.Tensor(data_type, (length, W), scope=tik.scope_ubuf, name="w_ub")

        result_ub = tik_instance.Tensor(data_type, (length, W),scope=tik.scope_ubuf, name="result_ub")
        exp_ub = tik_instance.Tensor(data_type, (length, W),scope=tik.scope_ubuf, name="exp_ub")
        reduce_ub = tik_instance.Tensor(data_type, (length,),scope=tik.scope_ubuf, name="reduce_ub")
        broadcast_ub = tik_instance.Tensor(data_type, (length, W),scope=tik.scope_ubuf, name="broadcast_ub")


        tik_instance.h_data_move(x_ub[:, :], x_gm[length * i: length * (i + 1), :])
        tik_instance.h_data_move(w_ub[:, :], y_gm[length * i: length * (i + 1), :])



        tik_instance.h_mul(exp_ub, x_ub, w_ub)
        tik_instance.h_reduce_max(reduce_ub, exp_ub, axis=(1))


        broadcast_scalar = tik_instance.Scalar(dtype=data_type)

        with tik_instance.for_range(0, length) as j:
            broadcast_scalar.set_as(reduce_ub[j])
            tik_instance.h_duplicate(broadcast_ub[j: j + 1, :], broadcast_scalar)

        tik_instance.h_sub(exp_ub, exp_ub, broadcast_ub)


        tik_instance.h_exp(exp_ub, exp_ub)
        tik_instance.h_duplicate(reduce_ub, 0.0)
        tik_instance.h_reduce_sum(reduce_ub, exp_ub, axis=(1))


        with tik_instance.for_range(0, length) as j:
            broadcast_scalar.set_as(reduce_ub[j])
            tik_instance.h_duplicate(broadcast_ub[j: j + 1, :], broadcast_scalar)


        tik_instance.h_div(result_ub, exp_ub, broadcast_ub)

        tik_instance.h_data_move(output_gm[length * i: length * (i + 1), :], result_ub[:, :])


    tik_instance.BuildCCE(kernel_name="softmax_tik", inputs=[x_gm, y_gm, ],
                          outputs=[output_gm, ])

    return tik_instance
