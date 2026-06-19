from pager import RowsModel, RegionModel, ImageModel
from pager.page_model.sub_models.dtype import Row, ImageSegment
from ...tokenizers import BaseTokenizer
import numpy as np
import matplotlib.pyplot as plt
class Ploter:
    def __init__(self, conf):
        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']
        self.loger.time_log()
        self.loger("Create Ploter")
        self.img_model = ImageModel()
        self.rows_model = RowsModel()
        self.regions_model = RegionModel()

    

    def set_dpi(self, dpi):
        plt.figure(dpi=dpi)
    
    def plot_img(self, img):
        self.img_model.img = img
        self.img_model.show()

    def plot_rows(self, array_json_rows):
        rows = [Row(json_row) for json_row in array_json_rows]
        for row in rows:
            row.segment.plot()


    def plot_tokens(self, tokenizer: BaseTokenizer , torch_dict, resize=None, markup=False):
        dict_features= tokenizer.get_dict_vec()  
        if resize is None:
            resize = (1, 1) 
        def get_coords(vec):
            rw, rh = resize
            x0 = vec[dict_features['x_top_left'][0]].cpu()
            x1 = vec[dict_features['x_bottom_right'][0]].cpu()
            y0 = vec[dict_features['y_top_left'][0]].cpu()
            y1 = vec[dict_features['y_bottom_right'][0]].cpu()
            h = vec[dict_features['height'][0]].cpu()
            w = vec[dict_features['width'][0]].cpu()
            return x0*rw, x1*rw, w*rw, y0*rh, y1*rh, h*rh

        ax = plt.gca()
        colors_alternative = [
            'gray',          # 1. Серый
            'tab:red',       # 2. Красный
            'tab:green',     # 3. Зеленый
            'tab:blue',      # 4. Синий
            'tab:orange',    # 5. Оранжевый
            'tab:purple',    # 6. Фиолетовый
            'tab:brown',     # 7. Коричневый
            'tab:pink',      # 8. Розовый
            'tab:olive',     # 9. Оливковый
            'tab:cyan',      # 10. Голубой
            'lime',          # 11. Лаймовый
            'teal',          # 12. Сине-зеленый
        ]

        if markup:
            inds= [int(np.argmax(e.cpu())) for e in torch_dict['true_nodes']]
        else:
            inds= [3 for _ in torch_dict["X"]]
        for vec, ind in zip(torch_dict["X"], inds):
            try:
                x0, x1, w, y0, y1, h = get_coords(vec)
                seg = ImageSegment(int(x0), int(y0), int(x1), int(y1)) 
               
                seg.plot(width=0.5, color=colors_alternative[ind])
                plt.scatter(x0+w/2, y0+h/2, color='b', marker='.')
            except:
                continue

        A = torch_dict['inds']
        if markup:
            inds= [e for e in torch_dict['true_edges']]
        else:
            inds= [True for _ in A[0]]
        for i, j, e in zip(A[0], A[1], inds):
            x0_0, x1_0, w_0, y0_0, y1_0, h_0 = get_coords(torch_dict["X"][i])
            x0_1, x1_1, w_1, y0_1, y1_1, h_1 = get_coords(torch_dict["X"][j])
            xc1 = (x0_0 + x1_0)/2
            yc1 = (y0_0 + y1_0)/2
            xc2 = (x0_1 + x1_1)/2
            yc2 = (y0_1 + y1_1)/2
            ax.plot([xc1, xc2], [yc1, yc2], 'g' if e else 'r', linewidth=0.7)
