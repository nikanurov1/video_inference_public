

def get_clas_conf(out_softmax, out_argmax):
    list_out =[]
    for row, index in zip(out_softmax, out_argmax):
        out={ "class":int(index),
                "conf":float(row[index]) }
        list_out.append(out)
    return list_out