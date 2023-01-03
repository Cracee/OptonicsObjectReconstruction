

def translate_pcd(point_cloud):
    raise NotImplementedError


def visualize_result(net_output, data_batch):
    for key in net_output.keys():
        print(key)
        item = net_output[key]
        print(len(item), end=" ")
        item = item[0]
        print(len(item), end=" ")
        item = item[0]
        print(item.shape)
    print("uwu")