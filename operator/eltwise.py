import math
import numpy 
from tbe import tik


def eltwise_tik():

    tik_instance = tik.Tik()

    

    input_x_gm = tik_instance.Tensor('float16', (16,16,16,16,16,), name="input_x_gm", scope=tik.scope_gm)
    input_y_gm = tik_instance.Tensor('float16', (16,16,16,16,16,), name="input_y_gm", scope=tik.scope_gm)
    output_z_gm = tik_instance.Tensor("float16", (16,16,16,16,16,), name="input_z_gm", scope=tik.scope_gm)

    input_x_ub = tik_instance.Tensor('float16',(63488,), name = "input_x_ub",scope = tik.scope_ubuf)
    input_y_ub = tik_instance.Tensor('float16',(63488,), name = "input_y_ub",scope = tik.scope_ubuf)

    move_num = 63488
    vector_mask_max = 8 * 16



    with tik_instance.for_range(0,2,block_num=2) as index:
        
        move_offset = index * 524288

        loop_time = 524288 // 63488

        with tik_instance.for_range(0, loop_time) as loop_index:
            move_offset = loop_index * 63488
            burst_len = math.ceil(63488 / 16)

            tik_instance.data_move(
                input_x_ub,
                input_x_gm[move_offset], 0, 1, burst_len, 0, 0

            )
            tik_instance.data_move(
                input_y_ub,
                input_y_gm[move_offset], 0, 1, burst_len, 0, 0
            )

            vadd_loop = move_num // (vector_mask_max * 255)
            add_offset = 0
            if vadd_loop > 0:
                with tik_instance.for_range(0, vadd_loop) as add_index:
                    add_offset = add_index * vector_mask_max * 255
                    tik_instance.vec_add(vector_mask_max,
                                         input_x_ub[add_offset],
                                         input_x_ub[add_offset],
                                         input_y_ub[add_offset],
                                         255, 8, 8, 8
                                         )
                add_offset = vadd_loop * vector_mask_max * 255
            repeat_time = (
                    move_num % (vector_mask_max * 255) // vector_mask_max
            )
            if repeat_time > 0:
                tik_instance.vec_add(
                    vector_mask_max,
                    input_x_ub[add_offset],
                    input_x_ub[add_offset],
                    input_y_ub[add_offset],
                    repeat_time, 8, 8, 8

                )
                add_offset += repeat_time * vector_mask_max
            last_num = move_num % vector_mask_max
            if last_num > 0:
                tik_instance.vec_add(
                    last_num,
                    input_x_ub[add_offset],
                    input_x_ub[add_offset],
                    input_y_ub[add_offset],
                    1, 8, 8, 8
                )

            tik_instance.data_move(output_z_gm[move_offset],
                                   input_x_ub, 0, 1, burst_len, 0, 0
                                   )

        
    tik_instance.BuildCCE(
        kernel_name="eltwise_tik",
        inputs=[input_x_gm, input_y_gm],
        outputs=[output_z_gm]
    )

    return tik_instance



if __name__ == '__main__':
    tik.instance = eltwise_tik()
    data_x = numpy.ones((16,16,16,16,16)).astype('float16')
    data_y = numpy.ones((16,16,16,16,16)).astype('float16')
    feed_dict = {'input_x_gm':data_x,'input_y_gm':data_y}
    model_data,=tik.instance.tikdb.start_debug(feed_dict=feed_dict,interactive=False)
    print(model_data)

