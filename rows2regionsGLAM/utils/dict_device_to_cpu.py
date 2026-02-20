def move_to_device(torch_dict, device):
    torch_dict_out = torch_dict.copy()
    
    tensor_keys = ['X', 'Y', 'sp_A', 'true_edges', 'true_nodes']
    
    for key in tensor_keys:
        if key in torch_dict_out:
            torch_dict_out[key] = torch_dict_out[key].to(device)
            
    return torch_dict_out


def get_dict_to_cpu(torch_dict_device):
    torch_dict_out = torch_dict_device.copy()
    
    tensor_keys = ['X', 'Y', 'sp_A', 'true_edges', 'true_nodes']
    
    for key in tensor_keys:
        if key in torch_dict_out:
            torch_dict_out[key] = torch_dict_out[key].cpu()
            
    return torch_dict_out