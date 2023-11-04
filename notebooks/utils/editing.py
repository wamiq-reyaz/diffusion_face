from PIL import Image
import numpy as np
from ipywidgets import widgets
from IPython.display import clear_output, display


# ----------------------------------------------
# create a random color_map
cmap = np.array([
    np.array((1.0, 1.0, 1.0), np.float32),
    np.array((255, 250, 79), np.float32) / 255.0,  # face
    np.array([255, 125, 138], np.float32) / 255.0,  # lb
    np.array([213, 32, 29], np.float32) / 255.0,  # rb
    np.array([0, 144, 187], np.float32) / 255.0,  # le
    np.array([0, 196, 253], np.float32) / 255.0,  # re
    np.array([255, 129, 54], np.float32) / 255.0,  # nose
    np.array([88, 233, 135], np.float32) / 255.0,  # ulip
    np.array([0, 117, 27], np.float32) / 255.0,  # llip
    np.array([255, 76, 249], np.float32) / 255.0,  # imouth
    np.array((1.0, 0.0, 0.0), np.float32),  # hair
    np.array((255, 250, 100), np.float32) / 255.0,  # lr
    np.array((255, 250, 100), np.float32) / 255.0,  # rr
    np.array((250, 245, 50), np.float32) / 255.0,  # neck
    np.array((0.0, 1.0, 0.5), np.float32),  # cloth
    np.array((0.0, 1.0, 0.1), np.float32),  # cloth
    np.array((0.5, 1.0, 0.1), np.float32),  # cloth
    np.array((1.0, 0.0, 0.5), np.float32),
    np.array((1.0, 1.0, 0.5), np.float32),
]
)
cmap = (cmap * 255).astype(np.uint8)

# ----------------------------------------------
class Viewer:
    def __init__(
        self,
        images,
        masks=None
    ):
        ''' images: list of images of shape (H, W, 3)
            masks: list of masks of shape (H, W)
        '''
        self.images = images
        self.masks = masks
        
        self.idx = 0
        self.overlay = not (self.masks is None)
        
        self.out = widgets.Output()
        self.button = widgets.ToggleButtons(
                options=['On', 'Off'],
                description='Overlay:',
                disabled=False,
                button_style='', # 'success', 'info', 'warning', 'danger' or ''
        )
        self.slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.images)-1,
            step=1,
            description='Frame:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
    
    def _on_button_clicked(self, b):
        if b['new'] == 'On':
            self.overlay = not (self.masks is None)
        else:
            self.overlay = False
            
        self._update()
        
    def _on_value_change(self, change):
        self.idx = change['new']
        self._update()
        
    def _update(self):
        with self.out:
            clear_output(wait=True)
            img = Image.fromarray(self.images[self.idx])
            if self.overlay:
                seg_map = self.masks[self.idx]
                _colored = cmap[seg_map.astype(np.int32)]
                _colored = _colored.astype(np.uint8)
                _colored = Image.fromarray(_colored)
                img = Image.blend(img, _colored, 0.5)
                
            clear_output(wait=True)
            display(img)
            
    def show(self):
        self.button.observe(self._on_button_clicked, names='value')
        self.slider.observe(self._on_value_change, names='value')
        self._update()
        
        display(self.button)
        display(self.slider)
        display(self.out)